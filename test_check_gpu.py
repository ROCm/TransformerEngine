import torch

major, minor = torch.cuda.get_device_capability()
print(f"GPU: gfx{major}{minor}")
print(f"Compute capability: {major}.{minor}")

if major == 9 and minor >= 5:
    print("This is MI350 (gfx950) - should use OCP FP8 formats")
    print("  e4m3fn, e5m2")
else:
    print("This is MI300/MI325 (gfx942) or older - should use NANOO FP8 formats")
    print("  e4m3fnuz, e5m2fnuz")
