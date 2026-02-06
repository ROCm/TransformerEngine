import argparse
import re
from pathlib import Path
from typing import List, Set

STRUCT_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^;]*)?;\s*$")
STRUCT_END_RE = re.compile(r"^\s*};\s*$")


def extract_fields_from_header(text: str, struct_name: str) -> List[str]:
    struct_start_re = re.compile(rf"\bstruct\s+{re.escape(struct_name)}\b")
    lines = text.splitlines()
    in_struct = False
    fields: List[str] = []
    buffer = ""
    for line in lines:
        if not in_struct:
            if struct_start_re.search(line):
                in_struct = True
            continue
        if STRUCT_END_RE.search(line):
            break
        # skip comments
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        line_no_comment = re.sub(r"//.*", "", line)
        buffer += " " + line_no_comment.strip()
        if ";" not in line_no_comment:
            continue
        m = STRUCT_FIELD_RE.search(buffer)
        if m:
            fields.append(m.group(1))
        buffer = ""
    return fields


def extract_usage_from_source(text: str, var_name: str) -> Set[str]:
    assign_re = re.compile(rf"\b{re.escape(var_name)}\.([A-Za-z_][A-Za-z0-9_]*)\b")
    return set(assign_re.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check aiter args usage vs header definition")
    parser.add_argument("--mode", choices=["fwd", "bwd"], required=True, help="Mode: fwd or bwd")
    args = parser.parse_args()

    header_path = Path(f"3rdparty/aiter/csrc/include/mha_{args.mode}.h")
    source_path = Path(f"transformer_engine/common/ck_fused_attn/src/ck_fused_attn_{args.mode}.cpp")
    header_text = header_path.read_text(encoding="utf-8")
    source_text = source_path.read_text(encoding="utf-8")

    header_fields = extract_fields_from_header(header_text, f"mha_{args.mode}_args")
    header_set = set(header_fields)
    used_fields = extract_usage_from_source(source_text, f"fmha_args")

    missing_in_usage = sorted(header_set - used_fields)
    unknown_in_header = sorted(used_fields - header_set)

    print(f"mha_{args.mode}_args fields in header:", len(header_set))
    print(f"mha_{args.mode}_args fields referenced in source:", len(used_fields))

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

    return 0


if __name__ == "__main__":
    main()
