import json
import os
import sys
import argparse

def extract_and_process_durations(input_file, output_file, kernel_keywords, num_warmup, num_iteration):
    with open(input_file, "r") as f:
        data = json.load(f)

    keyword_to_durations = {k: [] for k in kernel_keywords}

    for event in data.get("traceEvents", []):
        args = event.get("args", {})
        kernel_name = args.get("KernelName", "")
        duration = args.get("DurationNs")

        if duration is not None:
            for keyword in kernel_keywords:
                if keyword in kernel_name:
                    keyword_to_durations[keyword].append(int(duration))
                    break  # 防止同一个event被多个keyword重复统计

    output_lines = []

    for keyword in kernel_keywords:
        durations = keyword_to_durations[keyword]
        output_lines.append(f"== {keyword} ==")

        if not durations:
            output_lines.append("[无数据]")
            continue

        i = 0
        while i < len(durations):
            i += num_warmup  # 跳过warmup
            batch = []
            for _ in range(num_iteration):
                if i < len(durations):
                    batch.append(durations[i])
                    i += 1
            if batch:
                avg = sum(batch) / len(batch)
                output_lines.append(f"{avg:.2f}")
        output_lines.append("")  # 空行分隔

    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"已将所有 kernel 的平均耗时写入 {output_file}")

input_json = "/home/TransformerEngine/tests/cpp/build/results.json"
if len(sys.argv) > 1:
    output_txt = sys.argv[1]
else:
    output_txt = "/home/bwdprofiles/tmp/heyi.txt"

kernel_keywords = [
    "ln_fwd_",
    "ln_bwd_general_kernel",
    "ln_bwd_finalize"
]

num_warmup = 5
num_iteration = 10

extract_and_process_durations(input_json, output_txt, kernel_keywords, num_warmup, num_iteration)