#!/usr/bin/env python3
"""
脚本：针对指定 HIDDEN_SIZE/WTYPE/ITYPE/OTYPE/CTYPE，在 ln_fwd_cuda_kernel.cu 中批量替换 REGISTER_NORM_LAUNCHER 宏的
CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG 四个参数组合。
只替换匹配该前缀的行，保留其他注册宏不变。
"""
import re,os
import subprocess

# 需要替换的源文件路径
SOURCE_FILE = '/home/TransformerEngine/transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu'
RESULTS_DIR  = '/home/tuned_fwd/768/f16f16'
# 隐藏大小列表
hidden_sizes = [768]
# 构造前缀模板，format 时填入 hidden_size
PREFIX_TMPL = "REGISTER_NORM_LAUNCHER(LayerNorm, Forward, tuned, {hs}, fp16, fp16, fp16, fp32,"
# PREFIX_TMPL = "REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, {hs}, fp16, fp16, fp16, fp32,"

# # 要测试的参数组合
# ctas_per_row_list = [ 2] 
# warps_m_list      = [1]
# warps_n_list      = [8]
# bytes_per_ldg_list= [4,8,16,32]

ctas_per_row_list = [1] 
warps_m_list      = [2,1]
warps_n_list      = [2,4,8]
bytes_per_ldg_list= [8,16]
# 批量替换
for hs in hidden_sizes:
    # 每个 hidden_size 生成对应前缀
    prefix = PREFIX_TMPL.format(hs=hs)
    for ctas in ctas_per_row_list:
        for wm in warps_m_list:
            for wn in warps_n_list:
                for bpl in bytes_per_ldg_list:
                    if wm * wn < 2:
                        continue
                    lhs = hs // (bpl // 2)
                    rhs = ctas * wn * 32 * (lhs // (ctas * wn * 32))
                    # rhs = 1 * wn * 32 * (lhs // (1 * wn * 32))
                    if lhs != rhs:
                        continue
                    # if not (ctas == 1 or wm == 1):
                    #     continue
                    # 构造新的完整宏调用行
                    new_line = f"{prefix} {ctas}, {wm}, {wn}, {bpl});"#bwd
                    # 读取源文件
                    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    # 写回时替换匹配前缀的行
                    with open(SOURCE_FILE, 'w', encoding='utf-8') as f:
                        for line in lines:
                            if line.strip().startswith(prefix):
                                f.write(new_line + '\n')
                            else:
                                f.write(line)
                    print(f"Updated {SOURCE_FILE} for hidden_size={hs} with: WARPS_M={wm}, WARPS_N={wn}, BYTES_PER_LDG={bpl}")
                        
                    result=subprocess.run(['bash', './abs_do_fwd.sh'])
                    if result.returncode != 0:
                        print(f"Warning: abs_do.sh failed with exit code {result.returncode}")


proc = subprocess.run(
    ['python3', 'find_fast.py', RESULTS_DIR],
    stdout=subprocess.PIPE,
    text=True,
    check=True
)

best_fp = None
for line in proc.stdout.splitlines():
    if line.startswith('- ln_fwd_'):
        # 解析 “文件: /path/to/2048_12288_1_1_8_32”
        parts = line.split('文件:')
        if len(parts) == 2:
            best_fp = parts[1].strip()
        break

if not best_fp:
    print("Error: 没有找到最佳 ln_fwd_ 结果，退出。")
    sys.exit(1)

best_name = os.path.basename(best_fp)  # e.g. "2048_12288_1_1_8_32"
print("Best ln_fwd file:", best_name)

# —— 3. 从文件名拆出参数，并在 .cu 中替换宏行 —— #
tokens = best_name.split('_')
if len(tokens) != 6:
    print("Error: 无法解析文件名参数：", best_name)
    sys.exit(1)

hs2, n2, ctas2, wm2, wn2, bpl2 = tokens
prefix = PREFIX_TMPL.format(hs=hs2)
new_line = f"{prefix} {ctas2}, {wm2}, {wn2}, {bpl2});"

# 读源文件、替换所有匹配 prefix 的行
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(SOURCE_FILE, 'w', encoding='utf-8') as f:
    for line in lines:
        if line.strip().startswith(prefix):
            f.write(new_line + '\n')
        else:
            f.write(line)

print("已将所有前缀行替换为最佳组合：")
print("  ", new_line)