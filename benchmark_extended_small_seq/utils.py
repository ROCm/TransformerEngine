import jax
import jax.numpy as jnp
from einops import rearrange


def gen_data(dtype, n, seqlen_q, seqlen_kv, h, d, gqa_ratio: int = 1, nr_segments: int = 1):
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (n, seqlen_q, h, d), dtype=dtype) / 8
    k = jax.random.normal(key, (n, seqlen_kv, h // gqa_ratio, d), dtype=dtype) / 8
    v = jax.random.normal(key, (n, seqlen_kv, h // gqa_ratio, d), dtype=dtype) / 8
    do = jax.random.normal(key, (n, seqlen_q, h, d), dtype=dtype) / 8
    segment_ids_q = (
        jnp.array([range(0, nr_segments)], dtype=jnp.int32)
        .repeat(seqlen_q // nr_segments, axis=1)
        .repeat(n, axis=0)
    )
    segment_ids_kv = (
        jnp.array([range(0, nr_segments)], dtype=jnp.int32)
        .repeat(seqlen_kv // nr_segments, axis=1)
        .repeat(n, axis=0)
    )
    return q, k, v, do, segment_ids_q, segment_ids_kv


def segment_ids_to_cu_seqlens(segment_ids: jnp.ndarray, max_segments) -> jnp.ndarray:
    sid = segment_ids.astype(jnp.int32)
    counts = jax.vmap(lambda x: jnp.bincount(x, length=max_segments))(sid)
    lengths = counts.reshape(-1).astype(jnp.int32)
    return jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(lengths)], axis=0)


def jax_attention(query, key, value, segment_ids_q, segment_ids_kv, softmax_scale, is_causal, window_size):
    kv_heads = key.shape[-2]
    q_heads = query.shape[-2]
    if kv_heads != q_heads:
        key = jnp.repeat(key, q_heads // kv_heads, axis=-2)
        value = jnp.repeat(value, q_heads // kv_heads, axis=-2)

    query = jnp.einsum("b n h d -> b h n d", query)
    key = jnp.einsum("b n h d -> b h n d", key)
    scores = jnp.einsum("b h n d, b h s d -> b h n s", query, key)
    scores = (scores * softmax_scale).astype(query.dtype)

    mask_causal = (
        jnp.tril(jnp.ones((scores.shape[-1], scores.shape[-1])))
        if is_causal
        else jnp.ones((scores.shape[-2], scores.shape[-1]))
    )
    # mask is n,s,s where 1 means value is not masked which means that
    mask_seq = segment_ids_q[:, :, None] == segment_ids_kv[:, None, :]

    # only keep if not masked in both
    mask = jnp.logical_and(mask_causal, mask_seq)
    if window_size[0] != -1:
        mask_window = jnp.ones((scores.shape[-2], scores.shape[-1])) - jnp.tril(
            jnp.ones((scores.shape[-2], scores.shape[-1])), k=-window_size[0]
        )
    else:
        mask_window = jnp.ones((scores.shape[-2], scores.shape[-1]))

    mask = jnp.logical_and(mask, mask_window)

    scores = jnp.where(mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
    attention_weights = jax.nn.softmax(
        jnp.asarray(scores, dtype=jnp.float32), axis=-1
    )  # [batch, num_heads, seq_q, seq_kv]
    attention_weights = attention_weights.astype(value.dtype)

    out = jnp.einsum("b h s S, b S h d -> b h s d", attention_weights, value)
    out = rearrange(out, "b h n d -> b n h d")
    return out
