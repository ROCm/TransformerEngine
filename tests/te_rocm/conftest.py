# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Shared fixtures for the ROCm seam conformance tests (plugin plan P6).

These tests run against whatever `transformer_engine` resolves to - the fork's own tree or an
assembled overlay on PYTHONPATH - and against the built extension. They need the compiled
extension importable but do not launch kernels.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "proposals" / "te-rocm-plugin" / "tools"
MANIFEST = REPO / "proposals" / "te-rocm-plugin" / "divergence-manifest.yaml"
SUBMODULE = REPO / "3rdparty" / "transformer_engine_nvidia"
SEAM = "transformer_engine_torch"
COMPILED = "transformer_engine_rocm_torch"


def upstream_sha() -> str:
    import yaml
    return yaml.safe_load(MANIFEST.read_text())["metadata"]["upstream_sha"]


def upstream_py_files(sha: str) -> list[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", sha, "transformer_engine"],
                         capture_output=True, text=True, check=True, cwd=SUBMODULE).stdout
    return [l for l in out.splitlines() if l.endswith(".py")]


def upstream_source(sha: str, rel: str) -> str:
    return subprocess.run(["git", "show", f"{sha}:{rel}"], capture_output=True, text=True,
                          check=True, cwd=SUBMODULE).stdout


@pytest.fixture(scope="session")
def tex():
    import transformer_engine.pytorch  # noqa: F401  (installs the seam)
    return sys.modules[SEAM]


@pytest.fixture(scope="session")
def upstream_demand():
    """Static demand from upstream at the pin: names, enum members, and call-site shapes."""
    sha = upstream_sha()
    names: dict[str, set[str]] = {}          # name -> forms
    enum_members: dict[str, set[str]] = {}   # Enum -> members referenced
    calls: dict[str, list[tuple[int, set[str]]]] = {}  # fn -> [(n_positional, kwargs)]
    known_enums = {"DType", "NVTE_Bias_Type", "NVTE_Mask_Type", "NVTE_QKV_Format", "NVTE_QKV_Layout",
                   "NVTE_Softmax_Type", "NVTE_Fused_Attn_Backend", "CommOverlapType", "CommOverlapAlgo",
                   "Float8BlockScaleTensorFormat", "NVTERoutingMapFormat"}
    for rel in upstream_py_files(sha):
        try:
            tree = ast.parse(upstream_source(sha, rel))
        except SyntaxError:
            continue
        aliases = set(); from_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == SEAM: aliases.add(a.asname or SEAM)
            elif isinstance(node, ast.ImportFrom) and node.module == SEAM:
                for a in node.names:
                    if a.name != "*":
                        from_names.add(a.asname or a.name); names.setdefault(a.name, set()).add("from-import")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                names.setdefault(node.attr, set()).add("attr")
            # enum member refs: tex.Enum.MEMBER or Enum.MEMBER (from-imported)
            if isinstance(node, ast.Attribute):
                base = node.value
                if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in aliases and base.attr in known_enums:
                    enum_members.setdefault(base.attr, set()).add(node.attr)
                elif isinstance(base, ast.Name) and base.id in from_names and base.id in known_enums:
                    enum_members.setdefault(base.id, set()).add(node.attr)
            if isinstance(node, ast.Call):
                f = node.func; fname = None
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases:
                    fname = f.attr
                elif isinstance(f, ast.Name) and f.id in from_names:
                    fname = f.id
                if fname:
                    npos = sum(1 for a in node.args if not isinstance(a, ast.Starred))
                    has_star = any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords)
                    calls.setdefault(fname, []).append((-1 if has_star else npos, {k.arg for k in node.keywords if k.arg}))
    return {"sha": sha, "names": names, "enum_members": enum_members, "calls": calls}


def overlay_root() -> Path | None:
    import transformer_engine
    root = Path(transformer_engine.__file__).resolve().parent.parent
    return root if (root / "overlay-manifest.json").exists() else None
