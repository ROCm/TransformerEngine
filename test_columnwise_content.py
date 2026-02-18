"""
Understand what columnwise data actually contains.
Since it's not transposed, what's different about it?
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")
VEC_SIZE = 32

def test_columnwise_content():
    print("=" * 80)
    print("COLUMNWISE CONTENT ANALYSIS")
    print("=" * 80)

    M, K = 128, 256  # Must be divisible by 32
    torch.manual_seed(42)

    # Create a simple pattern to track
    # Each row has a different base value
    a_fp32 = torch.zeros((M, K), dtype=torch.float32, device=device)
    for i in range(M):
        a_fp32[i, :] = torch.arange(K, dtype=torch.float32, device=device) * 0.01 + i * 10.0

    print("\nOriginal matrix A:")
    print(f"a_fp32[0, :4] = {a_fp32[0, :4]}")  # Row 0, first 4 cols
    print(f"a_fp32[1, :4] = {a_fp32[1, :4]}")  # Row 1, first 4 cols
    print(f"a_fp32[:4, 0] = {a_fp32[:4, 0]}")  # Col 0, first 4 rows
    print(f"a_fp32[:4, 1] = {a_fp32[:4, 1]}")  # Col 1, first 4 rows

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32.to(torch.bfloat16))

    print("\n" + "-" * 80)
    print("Storage shapes:")
    print(f"Rowwise data: {a_mxfp8._rowwise_data.shape}")
    print(f"Rowwise scale: {a_mxfp8._rowwise_scale_inv.shape}")
    print(f"Columnwise data: {a_mxfp8._columnwise_data.shape}")
    print(f"Columnwise scale: {a_mxfp8._columnwise_scale_inv.shape}")

    # Convert to float for inspection
    row_data = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
    col_data = a_mxfp8._columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)

    print("\n" + "-" * 80)
    print("Rowwise data content:")
    print(f"row_data[0, :4] = {row_data[0, :4]}")
    print(f"row_data[1, :4] = {row_data[1, :4]}")
    print(f"row_data[:4, 0] = {row_data[:4, 0]}")

    print("\n" + "-" * 80)
    print("Columnwise data content:")
    print(f"col_data[0, :4] = {col_data[0, :4]}")
    print(f"col_data[1, :4] = {col_data[1, :4]}")
    print(f"col_data[:4, 0] = {col_data[:4, 0]}")

    print("\n" + "-" * 80)
    print("Scale analysis:")

    # Rowwise scales: [M, K//32]
    row_scale = a_mxfp8._rowwise_scale_inv
    print(f"\nRowwise scales shape: {row_scale.shape}")
    print(f"row_scale[0, :] = {row_scale[0, :]}")  # Scales for row 0
    print(f"row_scale[1, :] = {row_scale[1, :]}")  # Scales for row 1
    print("Meaning: Each row has {K//32} = 8 scale blocks")

    # Columnwise scales: [M//32, K]
    col_scale = a_mxfp8._columnwise_scale_inv
    print(f"\nColumnwise scales shape: {col_scale.shape}")
    print(f"col_scale[0, :4] = {col_scale[0, :4]}")  # Scales for first block of 32 rows
    print(f"col_scale[1, :4] = {col_scale[1, :4]}")  # Scales for second block of 32 rows
    print("Meaning: Each column has {M//32} = 4 scale blocks")

    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("\nColumnwise is NOT transposed data!")
    print("Instead, it's the SAME data quantized with DIFFERENT scales:")
    print("- Rowwise: quantizes each row with scales along columns")
    print("- Columnwise: quantizes each column with scales along rows")
    print("\nBoth have shape [M, K] but different quantization patterns!")

    # Verify this by dequantizing
    print("\n" + "-" * 80)
    print("Dequantization verification:")

    # Manual dequantization for rowwise (each row has its own scales)
    row_dequant = torch.zeros_like(a_fp32)
    for i in range(M):
        for j in range(K // VEC_SIZE):
            start = j * VEC_SIZE
            end = (j + 1) * VEC_SIZE
            scale = 2.0 ** (row_scale[i, j].to(torch.float32) - 127.0)
            row_dequant[i, start:end] = row_data[i, start:end] * scale

    # Manual dequantization for columnwise (each column has its own scales)
    col_dequant = torch.zeros_like(a_fp32)
    for i in range(M // VEC_SIZE):
        for j in range(K):
            start = i * VEC_SIZE
            end = (i + 1) * VEC_SIZE
            scale = 2.0 ** (col_scale[i, j].to(torch.float32) - 127.0)
            col_dequant[start:end, j] = col_data[start:end, j] * scale

    print(f"\nOriginal[0, :4] = {a_fp32[0, :4]}")
    print(f"Rowwise dequant[0, :4] = {row_dequant[0, :4]}")
    print(f"Columnwise dequant[0, :4] = {col_dequant[0, :4]}")

    print(f"\nOriginal[:4, 0] = {a_fp32[:4, 0]}")
    print(f"Rowwise dequant[:4, 0] = {row_dequant[:4, 0]}")
    print(f"Columnwise dequant[:4, 0] = {col_dequant[:4, 0]}")

test_columnwise_content()