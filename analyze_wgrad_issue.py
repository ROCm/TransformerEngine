"""
Analyze the wgrad scale issue.
"""

print("=" * 80)
print("WGRAD SCALE ANALYSIS")
print("=" * 80)

batch = 128
out_features = 1024
in_features = 768
VEC_SIZE = 32

print(f"\nOperation: dW = dY^T @ X")
print(f"  dY: [{batch}, {out_features}]")
print(f"  X:  [{batch}, {in_features}]")
print(f"  dW: [{out_features}, {in_features}]")

print("\n" + "-" * 80)
print("What we need for tl.dot_scaled:")
print(f"  dY^T: [{out_features}, {batch}] with scales [{out_features}, {batch//VEC_SIZE}]")
print(f"  X:    [{batch}, {in_features}] with scales [{batch//VEC_SIZE}, {in_features}]")

print("\n" + "-" * 80)
print("Option 1: Use dY rowwise and transpose")
print(f"  dY rowwise: [{batch}, {out_features}] with scales [{batch}, {out_features//VEC_SIZE}]")
print(f"  After transpose: [{out_features}, {batch}] with scales [{out_features//VEC_SIZE}, {batch}]")
print(f"  ✗ Scale shape wrong: [{out_features//VEC_SIZE}, {batch}] != [{out_features}, {batch//VEC_SIZE}]")

print("\n" + "-" * 80)
print("Option 2: Use dY columnwise and transpose")
print(f"  dY columnwise: [{batch}, {out_features}] with scales [{batch//VEC_SIZE}, {out_features}]")
print(f"  After transpose: [{out_features}, {batch}] with scales [{out_features}, {batch//VEC_SIZE}]")
print(f"  ✓ Scale shape correct!")

print("\n" + "-" * 80)
print("For X (second operand):")
print(f"  X columnwise: [{batch}, {in_features}] with scales [{batch//VEC_SIZE}, {in_features}]")
print(f"  ✓ This is exactly what we need!")

print("\n" + "=" * 80)
print("SOLUTION FOR WGRAD:")
print("  A (dY): Use columnwise and transpose")
print("  B (X):  Use columnwise (no transpose)")
print("=" * 80)