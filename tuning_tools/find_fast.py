#!/usr/bin/env python3
"""
脚本：遍历指定目录下所有文件，解析每个文件中
- ln_fwd_ kernel 的时间之和
- 将 ln_bwd_tuned_kernel 和 ln_bwd_finalize 两个 kernel 的时间之和合并为一个值
然后在所有文件中分别找出 ln_fwd_ 和合并后的 bwd 的最小值及对应文件，输出结果。
"""
import os
import sys
import re

def parse_file(filepath):
    """解析单个文件，返回 dict: 'ln_fwd_' -> sum, 'ln_bwd_total' -> combined sum"""
    sums = {}
    current = None
    times = []
    header_pat = re.compile(r"^==\s*(.+?)\s*==$")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current and times:
                    sums[current] = sum(times)
                    times = []
                continue
            m = header_pat.match(line)
            if m:
                current = m.group(1)
                times = []
            else:
                try:
                    times.append(float(line))
                except ValueError:
                    pass
        if current and times:
            sums[current] = sum(times)
    # 合并后两个 bwd kernels
    bwd_sum = sums.get('ln_bwd_tuned_kernel', 0) + sums.get('ln_bwd_finalize', 0)
    # 返回只有两项
    return {
        'ln_fwd_': sums.get('ln_fwd_', float('inf')),
        'ln_bwd_total': bwd_sum
    }

def find_minimums(dirpath):
    """遍历目录文件，返回 dict: key -> (min_sum, filepath)"""
    results = {}
    for name in os.listdir(dirpath):
        fp = os.path.join(dirpath, name)
        if not os.path.isfile(fp):
            continue
        file_sums = parse_file(fp)
        for key, val in file_sums.items():
            if key not in results or val < results[key][0]:
                results[key] = (val, fp)
    return results

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    d = sys.argv[1]
    if not os.path.isdir(d):
        print(f"Error: {d} is not a directory")
        sys.exit(1)
    mins = find_minimums(d)
    if not mins:
        print("No valid files found.")
        return
    print("最小时间和结果：")
    for key in ['ln_fwd_', 'ln_bwd_total']:
        val, fp = mins.get(key, (None, None))
        if val is None:
            print(f"- {key}: 无数据")
        else:
            print(f"- {key}: {val:.2f}    文件: {fp}")

if __name__ == '__main__':
    main()