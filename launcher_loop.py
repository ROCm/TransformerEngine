#!/usr/bin/env python3
"""
脚本：针对指定 HIDDEN_SIZE/WTYPE/ITYPE/OTYPE/CTYPE，在 ln_fwd_cuda_kernel.cu 中批量替换 REGISTER_NORM_LAUNCHER 宏的
CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG 四个参数组合。
只替换匹配该前缀的行，保留其他注册宏不变。
"""
import re
import subprocess

# 需要替换的源文件路径
SOURCE_FILE = '/home/TransformerEngine/transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu'

# 隐藏大小列表
hidden_sizes = [1024]
# 构造前缀模板，format 时填入 hidden_size
PREFIX_TMPL = "REGISTER_NORM_LAUNCHER(LayerNorm, Forward, tuned, {hs}, fp16, fp16, fp16, fp32,"

# # 要测试的参数组合
# ctas_per_row_list = [ 2] 
# warps_m_list      = [1]
# warps_n_list      = [4]
# bytes_per_ldg_list= [16,64]

ctas_per_row_list = [1,2] 
warps_m_list      = [1,4]
warps_n_list      = [1,4]
bytes_per_ldg_list= [4,8,16,32]
# 批量替换
for hs in hidden_sizes:
    # 每个 hidden_size 生成对应前缀
    prefix = PREFIX_TMPL.format(hs=hs)
    for ctas in ctas_per_row_list:
        for wm in warps_m_list:
            for wn in warps_n_list:
                for bpl in bytes_per_ldg_list:
                    if wm * wn != 4:
                        continue
                    lhs = hs // (bpl // 2)
                    rhs = ctas * wn * 32 * (lhs // (ctas * wn * 32))
                    if lhs != rhs:
                        continue
                    if not (ctas == 1 or wm == 1):
                        continue
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
                    print(f"Updated {SOURCE_FILE} for hidden_size={hs} with: CTAS_PER_ROW={ctas}, WARPS_M={wm}, WARPS_N={wn}, BYTES_PER_LDG={bpl}")
                    
                    subprocess.run(['bash', './abs_do.sh'], check=True)
