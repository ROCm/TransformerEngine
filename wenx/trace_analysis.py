import json
import sys
import numpy as np


path = sys.argv[1]

with open(path, 'r') as f:
    trace = json.load(f)

events = trace['traceEvents']

TARGTE_EVENTS_NAME = [
    'transpose_optimized_kernel(te_hip_fp8_e4m3 const*, float const*, te_hip_fp8_e4m3*, unsigned long, unsigned long)',
    '_ZN18transformer_engine12unary_kernelILi16ELb1EfNS_6detail5EmptyETnPFT1_S3_RKT2_EXadL_ZNS1_8identityEfRKS2_EE12hip_bfloat1615te_hip_fp8_e4m3EEvPKT4_PT5_PKS3_PS3_SK_S4_mm',
    'cast_transpose_optimized_kernel(hip_bfloat16 const*, float const*, te_hip_fp8_e4m3*, te_hip_fp8_e4m3*, float const*, float*, float*, unsigned long, unsigned long)',
    'cast_transpose_optimized_kernel(hip_bfloat16 const*, float const*, te_hip_fp8_e5m2*, te_hip_fp8_e5m2*, float const*, float*, float*, unsigned long, unsigned long)',
]


summary = {event:[] for event in TARGTE_EVENTS_NAME}
# summary = {}
for event in events:
    # if 'cast_transpose_optimized_kernel' in event['name']:
    if event['name'] in TARGTE_EVENTS_NAME:
        duration = event['dur']
        # if event['name'] not in summary:
        #     summary[event['name']] = []
        summary[event['name']].append(float(duration))

# import pdb; pdb.set_trace()

for k, v in summary.items():
    print(k)
    print(f"count: {len(v)}")
    print(f"dur on E2E: {np.array(v).sum()}")
    print(f"avg on E2E: {np.array(v).sum()/len(v)}")
    if True or "cast_transpose" in k:
        print(f"{v}")
    # print(f"details: {np.array(v)}")
    print("==========================================")
