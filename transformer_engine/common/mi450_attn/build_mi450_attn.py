# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Build the attention library of 3rdparty/te-mi450-attn-kernels out of tree, for TE.

    python build_mi450_attn.py --root <te-mi450-attn-kernels> --arch gfx1250 --build-dir DIR --namespace NS

Runs the library's own build (``kernels/build.py``, which rebuilds only what is out of date)
into ``DIR`` with every exported symbol named ``<NS>_<name>``, so the library lands at
``DIR/lib/<NS>/libattn_<NS>.so``.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    os.environ["ATTN_NAMESPACE"] = args.namespace
    sys.path.insert(0, str(args.root.resolve()))
    from kernels.build import BuildError, library  # pylint: disable=import-outside-toplevel

    try:
        lib = library(args.arch, args.build_dir.resolve())
    except (BuildError, FileNotFoundError) as e:
        print(f"MI450 attention library build failed: {e}", file=sys.stderr)
        return 1
    print(f"MI450 attention library: {lib.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
