# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.


"""
This script is run during setup through setup.py, and can be run independently
to check that the fields defined in the mha_{fwd,bwd}_args structs in the AITER
headers are correctly referenced in the source code.
"""

import argparse
import re
from pathlib import Path
from typing import List, Set
import sys

def parse_with_skip_comments(buffer, line, regex, outputs):
    # skip comments
    stripped = line.strip()
    if not stripped or stripped.startswith("//"):
        return
    line_no_comment = re.sub(r"//.*", "", line)
    buffer[0] += " " + line_no_comment.strip()
    if ";" not in line_no_comment:
        return
    match = regex.search(buffer[0])
    if match:
        outputs.append(match.group(1))
    buffer[0] = ""


def extract_fields_from_header(text: str, struct_name: str) -> List[str]:
    struct_field_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^;]*)?;\s*$")
    struct_end_re = re.compile(r"^\s*};\s*$")

    struct_start_re = re.compile(rf"\bstruct\s+{re.escape(struct_name)}\b")
    lines = text.splitlines()
    in_struct = False
    fields: List[str] = []
    buffer = [""]
    for line in lines:
        if not in_struct:
            if struct_start_re.search(line):
                in_struct = True
            continue
        if struct_end_re.search(line):
            break
        parse_with_skip_comments(buffer, line, struct_field_re, fields)
    return fields


def extract_usage_from_source(text: str, var_name: str) -> Set[str]:
    assign_re = re.compile(rf"\b{re.escape(var_name)}\.([A-Za-z_][A-Za-z0-9_]*)\b\s*=")
    assignments = []
    lines = text.splitlines()
    buffer = [""]
    for line in lines:
        parse_with_skip_comments(buffer, line, assign_re, assignments)
    return set(assignments)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check aiter args usage vs header definition")
    parser.add_argument("--mode", choices=["fwd", "bwd", "both"], default="both", help="Mode: fwd, bwd, or both")
    parser.add_argument("--te-dir", type=Path, default=Path(__file__).parent.parent, help="Root directory of TransformerEngine")
    args = parser.parse_args()
    modes = ["fwd", "bwd"] if args.mode == "both" else [args.mode]
    mismatch = 0
    for mode in modes:
        header_path = args.te_dir / f"3rdparty/aiter/csrc/include/mha_{mode}.h"
        source_path = args.te_dir / f"transformer_engine/common/ck_fused_attn/src/ck_fused_attn_{mode}.cpp"
        header_text = header_path.read_text(encoding="utf-8")
        source_text = source_path.read_text(encoding="utf-8")

        header_fields = extract_fields_from_header(header_text, f"mha_{mode}_args")
        header_set = set(header_fields)
        used_fields = extract_usage_from_source(source_text, f"fmha_args")

        missing_in_usage = sorted(header_set - used_fields)
        unknown_in_header = sorted(used_fields - header_set)
        mismatch += len(missing_in_usage) + len(unknown_in_header)

        print(f"\nAnalyzing mha_{mode}_args\n")
        print(f"mha_{mode}_args fields in header:", len(header_set))
        print(f"mha_{mode}_args fields referenced in source:", len(used_fields))

        if missing_in_usage:
            print("\nFields present in header but not referenced in source:")
            for name in missing_in_usage:
                print(f"  - {name}")
        else:
            print("\nAll header fields are referenced in source.")

        if unknown_in_header:
            print("\nFields referenced in source but not in header:")
            for name in unknown_in_header:
                print(f"  - {name}")
        else:
            print("\nNo unknown fields referenced in source.")

    if mismatch:
        print(f"\nTotal mismatched fields: {mismatch}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
