import pandas as pd
import panel as pn
import seaborn as sns
from matplotlib.figure import Figure
from jax import numpy as jnp

pn.extension(design="material", sizing_mode="stretch_width")

ATTRIBUTES = [
    "bias_config",
    "attn_mask_type",
    "qkv_layout",
    "is_training",
    "swa",
    "dropout",
    "mode",
    "dtype",
    "seq_desc_format",
]
CONVERTERS = {
    "attn_mask_type": {
        "AttnMaskType.NO_MASK": "None",
        "AttnMaskType.CAUSAL_MASK": "Causal",
        "AttnMaskType.PADDING_MASK": "Padding",
        "AttnMaskType.PADDING_CAUSAL_MASK": "Padding Causal",
        "AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK": "Causal Bottom-Right",
        "AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK": "Padding Causal Bottom-Right",
    },
    "attn_bias_type": {
        "AttnBiasType.NO_BIAS":"None",
        "AttnBiasType.POST_SCALE_BIAS":"Post-Scale Bias",
    },
    "dropout": {
        0: False,
        0.1: True,
    },
    "dtype": {
        str(jnp.float16):"FP16",
        str(jnp.bfloat16):"BF16",
    },
    "qkv_layout": {
        "QKVLayout.BS3HD":"BSHD-Packed",
        "QKVLayout.BSHD_BS2HD":"BSHD-KV-Packed",
        "QKVLayout.BSHD_BSHD_BSHD":"BSHD-Separate",
        "QKVLayout.T3HD":"THD-Packed",
        "QKVLayout.THD_T2HD":"THD-KV-Packed",
        "QKVLayout.THD_THD_THD":"THD-Separate",
    },
    "bias_shape":{
        "BiasShape._1HSS":"_1HSS",
        "NaN":"None",
    },
    "seq_desc_format":{
        "SeqDescFormat.Mask": "Mask",
        "SeqDescFormat.SegmentIDs": "SegmentIDs",
        "SeqDescFormat.Seqlens": "Seqlens",
    }
}
BIAS_CONFIGS = {
    ("None", "None"): "None",
    ("Post-Scale Bias", "_1HSS"): "Post-Scale _1HSS"
}

@pn.cache
def get_data():
    df = pd.read_csv("output_main.csv").fillna("NaN")
    df["time"] *= 1000
    for key in CONVERTERS:
        df[key] = df[key].map(lambda x: CONVERTERS[key][x])
    df["bias_config"] = df.apply(
        lambda row: BIAS_CONFIGS[(row["attn_bias_type"], row["bias_shape"])],
        axis=1
    )
    df = df.drop(columns=["attn_bias_type", "bias_shape"])
    return df

def _selector_widgets():
    df = get_data().fillna("NaN")
    return {
        cat: pn.widgets.Select(name=cat, options=list(df[cat].unique()))
        for cat in ATTRIBUTES
    }


selector_widgets = _selector_widgets()

def make_plot(hue, indep, percentile, **kwargs):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = get_data()
    for attr in ATTRIBUTES:
        if attr not in {hue, indep}:
            df = df[(df[attr]==kwargs[attr])]

    for idx in df[indep].unique():
        for jdx in df[hue].unique():
            subset = df[(df[indep]==idx) & (df[hue]==jdx)]
            df[(df[indep]==idx) & (df[hue]==jdx)] = subset[subset.time < subset.time.quantile(percentile)]

    if not df.empty:
        ax.set(xlabel=indep, ylabel='Time (ms)')
        sns.swarmplot(ax=ax, data=df, x=indep, y="time", hue=hue, dodge=True)
        return fig

hue_selector = pn.widgets.Select(name="Hue", options=ATTRIBUTES, value="dtype")
indep_selector = pn.widgets.Select(name="Independent Variable", options=ATTRIBUTES, value="attn_mask_type")
percentile_trim = pn.widgets.FloatSlider(value=.95, start=0, end=1, step=.01, name="Percentile Trim")
bound_make_plot = pn.bind(
    make_plot,
    hue=hue_selector,
    indep=indep_selector,
    percentile=percentile_trim,
    **selector_widgets,
)

template = pn.template.BootstrapTemplate(
    title='JAX Fused Attention Benchmarks',
    sidebar=pn.Row(
        pn.Column(hue_selector, indep_selector, percentile_trim),
        pn.Column(*[selector_widgets[k] for k in selector_widgets]),
    )
)
template.main.append(pn.pane.Matplotlib(bound_make_plot, dpi=144, height=600))
template.servable();