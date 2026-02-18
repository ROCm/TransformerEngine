import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Test with various dimensions
test_cases = [
    (128, 256, 512),  # Original failing case
    (64, 128, 256),   # Smaller test
    (256, 512, 128),  # Different aspect ratio
]

for M, N, K in test_cases:
    print("=" * 60)
    print(f"Testing M={M}, N={N}, K={K}")
    print("=" * 60)

    torch.manual_seed(42)
    a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    print(f"A shape: {a_mxfp8.size()}")
    print(f"B shape: {b_mxfp8.size()}")
    print(f"Expected output shape: [{M}, {N}]")

    try:
        output = te_generic_gemm_triton(
            A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
            quantizer=None, output_dtype=tex.DType.kBFloat16,
            bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
            gelu=False, gelu_in=torch.Tensor(), grad=False,
            workspace=torch.Tensor(), workspaceSize=0,
            accumulate=False, use_split_accumulator=False,
            comm_overlap=False, comm_type=0,
            extra_output=torch.Tensor(), bulk_overlap=False,
        )[0]

        print(f"✓ Output shape: {output.shape}")

        # Check for inf/nan
        if torch.any(torch.isinf(output)) or torch.any(torch.isnan(output)):
            print(f"✗ Output contains inf/nan!")
        else:
            # Check accuracy
            ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
            max_diff = torch.max(torch.abs(output - ref)).item()
            print(f"Max difference: {max_diff:.4f}")
            if max_diff < 10.0:
                print(f"✓ Reasonable accuracy")
            else:
                print(f"✗ Large error")

    except Exception as e:
        print(f"✗ Error: {e}")

    print()