"""
Add more detailed shape debugging to understand the issue.
"""

import os

# Add this to the gemm_triton.py file at the beginning of te_generic_gemm_triton
debug_code = '''
    # Debug shapes at entry
    import os
    if os.getenv("DEBUG_MXFP8_SHAPES"):
        print(f"\\n[SHAPE DEBUG] te_generic_gemm_triton entry:")
        print(f"  A shape: {A.shape if hasattr(A, 'shape') else A.size() if hasattr(A, 'size') else 'unknown'}")
        print(f"  B shape: {B.shape if hasattr(B, 'shape') else B.size() if hasattr(B, 'size') else 'unknown'}")
        print(f"  transa={transa}, transb={transb}")
        print(f"  grad={grad}")
'''

# Let's add this debug code to our implementation
import transformer_engine.pytorch.gemm_triton as gemm_triton
import inspect

# Get the source
source = inspect.getsource(gemm_triton.te_generic_gemm_triton)

# Find where to insert (after the function definition)
lines = source.split('\n')
for i, line in enumerate(lines):
    if 'def te_generic_gemm_triton' in line:
        # Find the end of the function signature
        j = i
        while not lines[j].strip().endswith(':'):
            j += 1
        # Insert debug code after the function signature
        indent = '    '
        debug_lines = [indent + line for line in debug_code.strip().split('\n')]
        lines = lines[:j+1] + debug_lines + lines[j+1:]
        break

# Reconstruct the function
new_source = '\n'.join(lines)

# Create a new function with our debug code
exec(new_source, gemm_triton.__dict__)

print("Debug code added to te_generic_gemm_triton")
print("Set DEBUG_MXFP8_SHAPES=1 to see shape debug output")