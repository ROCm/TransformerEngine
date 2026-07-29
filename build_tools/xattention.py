# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""xAttention (fp8 flash-attn) build integration for TransformerEngine (ROCm).

xAttention ships its open-source layers (``interface``, ``codeGen``) as source
plus a prebuilt, closed-source core (``lib/<arch>/xattention_tileiras.a``) per
GPU architecture. This module drives xAttention's own CMake to produce the
open-source static archives and the generated ``Config.h``, then exposes a
setuptools ``CppExtension`` (``transformer_engine_xattention``) that links them
against the prebuilt core.

The extension is a *second*, self-contained module built alongside
``transformer_engine_torch`` (it needs ``-std=c++20`` and a static-archive link
group that would otherwise contaminate the main extension). It is only added to
the build when the submodule is checked out and the target GPU arch is one
xAttention ships (gfx950 -> mi350, gfx1250 -> mi450); otherwise the build skips
it gracefully and the runtime wrapper reports the backend as unavailable.

The backend additionally requires torch >= 2.10, which the rest of TransformerEngine
does not; see ``torch_is_supported``.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import rocm_path

# gfx target -> xAttention arch dir (the two arches xAttention ships a core for).
_GFX_TO_XATTN_ARCH = {
    "gfx950": "mi350",
    "gfx1250": "mi450",
}

# Kernel build mode. AOT (default) prebuilds all kernels at build time (no runtime
# SP3/codegen dependency, no first-call latency); JIT generates them at runtime and
# is intended for development. Dev opt-in: NVTE_XATTENTION_JIT=1 or
# NVTE_XATTENTION_KERNEL_MODE=JIT.
_DEFAULT_KERNEL_MODE = "AOT"

# Minimum torch for the xAttention backend; see ``torch_is_supported``.
_MIN_TORCH_VERSION = (2, 10)


def kernel_mode() -> str:
    """``"AOT"`` (default) or ``"JIT"`` (development opt-in)."""
    if os.getenv("NVTE_XATTENTION_JIT", "0") != "0":
        return "JIT"
    mode = os.getenv("NVTE_XATTENTION_KERNEL_MODE", _DEFAULT_KERNEL_MODE).upper()
    return mode if mode in ("AOT", "JIT") else _DEFAULT_KERNEL_MODE


def xattention_source_dir() -> Optional[Path]:
    """Path to the xAttention checkout, or ``None`` if not provisioned.

    Honors ``NVTE_XATTENTION_SOURCE_DIR`` (dev override) and otherwise falls
    back to the in-tree submodule at ``3rdparty/xAttention``. Returns ``None``
    unless the checkout actually contains xAttention's CMake project.
    """
    override = os.getenv("NVTE_XATTENTION_SOURCE_DIR")
    if override:
        xa = Path(override).expanduser().resolve()
    else:
        # build_tools/xattention.py -> repo root -> 3rdparty/xAttention
        xa = (Path(__file__).parent.parent / "3rdparty" / "xAttention").resolve()
    if (xa / "CMakeLists.txt").is_file() and (xa / "csrc").is_dir():
        return xa
    return None


def _candidate_gfx_archs() -> List[str]:
    """Ordered gfx/arch candidates for the target GPU.

    Priority: explicit override, then the physical device (the reliable runtime
    signal), then ``PYTORCH_ROCM_ARCH`` — which is often a *multi-arch build
    list* whose first entry need not be the device, so we scan the whole list
    for a supported arch rather than taking the first.
    """
    cands: List[str] = []

    override = os.getenv("NVTE_XATTENTION_ARCH")
    if override:
        cands.append(override.strip())

    # Physical device(s) — most reliable for "what will actually run".
    for tool, is_enum in ((["rocm_agent_enumerator"], True), (["rocminfo"], False)):
        try:
            out = subprocess.run(tool, capture_output=True, text=True, check=False).stdout
        except (FileNotFoundError, OSError):
            continue
        found = False
        for line in out.splitlines():
            s = line.strip()
            if is_enum and s.startswith("gfx"):
                cands.append(s)
                found = True
            elif not is_enum and s.startswith("Name:") and "gfx" in s:
                cands.append(s.split()[-1])
                found = True
        if found:
            break

    # Build-target list (may target many arches at once).
    for env in ("PYTORCH_ROCM_ARCH", "HCC_AMDGPU_TARGET"):
        val = os.getenv(env)
        if val:
            cands += [a.strip() for a in val.replace(",", ";").split(";") if a.strip()]

    return cands


def target_xattn_arch() -> Optional[str]:
    """xAttention arch dir (``mi350``/``mi450``) for the target GPU, or ``None``.

    Returns the first candidate that maps to (or already is) a supported
    xAttention arch. ``None`` means "no xAttention-supported arch selected" and
    the caller should skip the extension.
    """
    for cand in _candidate_gfx_archs():
        if cand in _GFX_TO_XATTN_ARCH.values():  # already an xAttention arch name
            return cand
        mapped = _GFX_TO_XATTN_ARCH.get(cand)
        if mapped is not None:
            return mapped
    return None


def torch_is_supported() -> bool:
    """Whether the installed torch is new enough for the xAttention backend.

    The closed core references ``c10::TensorImpl::{incref,decref}_pyobject``,
    which torch only defines from 2.10 on. Older torch cannot link the extension
    without stubbing those symbols out, and stubbing them is not safe: they are
    ``override final`` on ``TensorImpl``, so at -O3 the compiler devirtualizes
    ``intrusive_ptr``'s calls into direct references that bind to any local
    definition. Only the decrefs devirtualize into our objects (the matching
    increfs stay inside libtorch), so a stub silently drops torch's PyObject
    refcounting and retains the wrapper -- and the storage -- of every tensor
    crossing the boundary. Requiring 2.10 keeps that failure mode impossible.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    try:
        major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    except ValueError:
        # Unparseable version (custom build); assume new enough and let the link
        # speak for itself rather than skipping the backend silently.
        return True
    return (major, minor) >= _MIN_TORCH_VERSION


def xattention_enabled() -> bool:
    """Whether to build the xAttention extension in this build.

    Requires an opt-out-able toggle (``NVTE_BUILD_XATTENTION``, default on when
    provisioned), a checked-out source tree, a supported target arch, and a
    supported torch (see ``torch_is_supported``).
    """
    if os.getenv("NVTE_BUILD_XATTENTION", "1") == "0":
        return False
    if xattention_source_dir() is None or target_xattn_arch() is None:
        return False
    if not torch_is_supported():
        print(
            "Skipping xAttention extension: requires torch >= "
            f"{'.'.join(str(part) for part in _MIN_TORCH_VERSION)}."
        )
        return False
    return True


def _build_core(xa_dir: Path, arch: str, mode: str) -> Tuple[Path, Path, Optional[Path]]:
    """Drive xAttention's CMake to build ``interface``/``codeGen`` (+ AOT kernels).

    Returns ``(build_lib_dir, build_include_dir, kernel_base_dir)`` where
    ``kernel_base_dir`` is the parent of the ``<arch>`` kernel dir for AOT
    builds (the runtime appends ``/<arch>`` itself) and ``None`` for JIT. The
    build tree lives at ``<xa_dir>/build`` so the generated ``Config.h`` lands
    at the conventional ``build/include/Config.h``. CMake is incremental, so
    re-running on an unchanged tree is cheap.
    """
    build_dir = xa_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(xa_dir),
        "-B",
        str(build_dir),
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGPU_ARCH={arch}",
        # AOT prebuilds .co kernels at build time (no runtime SP3/codegen); JIT
        # generates them at runtime (the wrapper points at a writable dir).
        f"-DBUILD_KERNEL={mode}",
        # The assembly-text helper is only needed by xAttention's own unit tests.
        "-DXATT_ENABLE_CODE_INSPECTION=OFF",
    ]
    subprocess.run(configure, check=True, cwd=str(xa_dir))

    targets = ["interface", "codeGen"]
    build_env = os.environ.copy()
    if mode == "AOT":
        # Populates <build>/kernels/<arch> with .co artifacts via the generator.
        targets.append("prebuild_kernels")
        # The generator runs at build time and needs the SP3 assembler + a
        # writable scratch dir; the closed core otherwise uses packaging-machine
        # absolute paths that don't exist here.
        sp3_dir = xa_dir / "sp3" / arch
        tmp_dir = build_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        build_env.setdefault("XATT_SP3_DIR", str(sp3_dir))
        build_env.setdefault("XATT_TMP_DIR", str(tmp_dir))
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", *targets],
        check=True,
        cwd=str(xa_dir),
        env=build_env,
    )

    build_lib = build_dir / "lib"
    build_include = build_dir / "include"
    for name in ("libinterface.a", "libcodeGen.a"):
        if not (build_lib / name).is_file():
            raise RuntimeError(
                f"xAttention build did not produce {build_lib / name}; "
                "check the CMake output above."
            )

    kernel_base: Optional[Path] = None
    if mode == "AOT":
        kernel_base = build_dir / "kernels"
        arch_kernels = kernel_base / arch
        if not arch_kernels.is_dir() or not any(arch_kernels.glob("*.co")):
            raise RuntimeError(
                f"AOT kernel generation produced no .co files under {arch_kernels}; "
                "check the prebuild_kernels output above (or build with "
                "NVTE_XATTENTION_JIT=1 for a JIT development build)."
            )
    return build_lib, build_include, kernel_base


def _write_paths_module(
    xa_dir: Path, arch: str, mode: str, kernel_base: Optional[Path]
) -> None:
    """Record the xAttention data root, arch, and kernel mode for the wrapper.

    The wrapper reads this to point the closed core at its SP3 toolchain and,
    for AOT builds, at the prebuilt kernel dir — resolving locations the runtime
    can't otherwise infer (e.g. non-editable installs). Best-effort.
    """
    dst = (
        Path(__file__).parent.parent
        / "transformer_engine"
        / "pytorch"
        / "attention"
        / "dot_product_attention"
        / "_xattention_paths.py"
    )
    kernel_repr = f'r"{kernel_base}"' if kernel_base is not None else "None"
    contents = (
        "# Generated by build_tools/xattention.py at build time. Do not edit.\n"
        f'XATTENTION_ROOT = r"{xa_dir}"\n'
        f'XATTENTION_ARCH = "{arch}"\n'
        f'XATTENTION_KERNEL_MODE = "{mode}"\n'
        f"XATTENTION_KERNEL_DIR = {kernel_repr}\n"
    )
    try:
        dst.write_text(contents)
    except OSError:
        pass


def setup_xattention_extension(csrc_source_files) -> "object":
    """Build the xAttention core and return the ``transformer_engine_xattention``
    CppExtension. Caller must have confirmed :func:`xattention_enabled`.
    """
    from torch.utils.cpp_extension import CppExtension  # local: torch may be absent

    xa_dir = xattention_source_dir()
    arch = target_xattn_arch()
    assert xa_dir is not None and arch is not None, "xattention_enabled() not checked"

    core_a = xa_dir / "lib" / arch / "xattention_tileiras.a"
    if not core_a.is_file():
        raise RuntimeError(
            f"Prebuilt xAttention core not found: {core_a}\n"
            f"The submodule ships a core only for its packaged arches "
            f"({', '.join(sorted(_GFX_TO_XATTN_ARCH.values()))})."
        )

    mode = kernel_mode()
    build_lib, build_include, kernel_base = _build_core(xa_dir, arch, mode)
    interface_a = build_lib / "libinterface.a"
    codegen_a = build_lib / "libcodeGen.a"

    _write_paths_module(xa_dir, arch, mode, kernel_base)

    rocm_home, _ = rocm_path()
    rocm_home = Path(rocm_home)

    binding_dir = Path(csrc_source_files) / "xattention"
    sources = [str(binding_dir / "xattention_binding.cpp")]

    include_dirs = [
        str(xa_dir / "include"),
        str(build_include),
        str(rocm_home / "include"),
    ]

    cxx_flags = ["-O3", "-std=c++20", "-D__HIP_PLATFORM_AMD__", "-fvisibility=hidden"]

    # The open-source archives and the closed core are mutually recursive
    # (interface -> codeGen -> core and back), so wrap them in a link group
    # rather than depending on link order.
    link_args = [
        "-Wl,--start-group",
        str(interface_a),
        str(codegen_a),
        str(core_a),
        "-Wl,--end-group",
        f"-L{rocm_home / 'lib'}",
        "-lamdhip64",
        f"-Wl,-rpath,{rocm_home / 'lib'}",
    ]

    return CppExtension(
        name="transformer_engine_xattention",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=link_args,
    )
