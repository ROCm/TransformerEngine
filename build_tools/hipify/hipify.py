# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Union, Optional


def do_hipify(te_root: Union[Path, str], src_dir: Union[Path, str],
              common_headers_dir: Optional[Union[Path, str]] = None,
              result_file: Optional[str] = None) -> dict:
    """
    Converts TransformerEngine CUDA code to HIP code using the hipify_torch module.
    This function runs the hipify transformation on source files in the specified TE directory,
    converting CUDA-specific code to HIP-compatible code. It can optionally save the
    transformation results to a JSON file.
    Args:
        te_root (Union[Path, str]): TE project root directory.
            Used to located build tools and the hipify_torch module.
        src_dir (Union[Path, str]): Source directory containing CUDA files to be hipified.
            The hipified output will be written to the same directory.
        common_headers_dir (Optional[Union[Path, str]]): directory containing common header
            If not set {te_root}/transformer_engine will be used as default.
        result_file (Optional[str]): Path to an optional JSON file where hipify results
            will be saved. If not set, results are not persisted to disk.
    Returns:
        dict: A dictionary containing the hipify transformation results, including
            details about converted files and any transformations applied.
    Raises:
        ImportError: If the hipify_torch module cannot be imported from the project root.
        FileNotFoundError: If the source directory or required configuration files do not exist.
        IOError: If there is an error writing to the result_file.
    """
    te_root = Path(te_root).resolve()
    hipify_root = te_root / "3rdparty" / "hipify_torch"
    sys.path.insert(0, str(hipify_root))
    from hipify_torch.v2 import hipify_python as hipify_module 

    common_headers_dir = (Path(common_headers_dir).resolve() if common_headers_dir else
                          te_root / "transformer_engine")
    include_dirs = [common_headers_dir,
                    common_headers_dir / "common",
                    common_headers_dir / "common" / "include",
                    Path(src_dir).resolve()]

    print(f"Run hipify on {src_dir}")

    hipify_result = hipify_module.hipify(
        project_directory=src_dir,
        output_directory=src_dir,
        includes=["*/common/*", str(Path(src_dir)/"*")],
        ignores=["*/amd_detail/*", "*/aotriton/*", "*/ck_fused_attn/*", "*/rocshmem_api/*"],
        header_include_dirs=include_dirs,
        custom_map_list= te_root / "build_tools" / "hipify" / "custom_map.json",
        extra_files=[],
        is_pytorch_extension=True,
        hipify_extra_files_only=False,
        show_detailed=False,
        no_math_replace=True)
    
    # Convert hipify objects to dictionaries for consistent behavior
    hipify_result = {k: v.asdict() if hasattr(v, 'asdict') else v for k, v in hipify_result.items()}

    if result_file:
        with open(result_file, 'w') as dict_file:
            dict_file.write(json.dumps(hipify_result))

    return hipify_result


def get_hipified_sources(hipify_result: Union[str, dict], sources: Union[list[Union[Path, str]], Path, str],
                         src_base_path: Union[Path, str]) -> Union[list[str], str]:
    """
    Process and return hipified source file paths, updating the source list file if provided.
    
    This function takes hipify conversion results and a list of source files, then returns
    the corresponding hipified file paths relative to the source base directory. If the sources
    parameter points to a file, the file is updated with the hipified paths and the file path
    is returned. Otherwise, a list of hipified paths is returned.
    
    Args:
        hipify_result (Union[str, dict]): Either a file path to a JSON file containing hipify
            conversion results as a dictionary, or a dictionary directly mapping original file
            paths to their hipification results. Each result should have a `hipified_path`
            attribute indicating the converted file path.
        sources (Union[list[Union[Path, str]], Path, str]): Either a list of source file paths
            (as strings or Path objects), or a file path (as string or Path) containing one
            source file path per line. These are the original CUDA source files to be hipified.
        src_base_path (Union[Path, str]): The base directory path used to compute relative
            paths for the output. All returned paths will be relative to this directory.
    
    Returns:
        Union[list[str], str]: If `sources` is a file path, returns the file path after updating
            it with hipified source paths. If `sources` is a list, returns a list of strings
            representing relative paths to hipified source files. Duplicate entries are removed
            by converting to a set internally.
    """
    if isinstance(hipify_result, str):
        with open(hipify_result, 'r') as dict_file:
            hipify_result = json.load(dict_file)
    else:
        hipify_result = dict(hipify_result)

    sources_fname = None
    if isinstance(sources, (str, Path)):
        sources_fname = os.path.abspath(str(sources))
        sources =  [line.strip() for line in open(sources_fname).readlines() if line.strip()]

    # Because hipify output_directory == project_directory
    # Original sources list may contain previous hipifying results that ends up with duplicated entries
    # Keep unique entries only
    hipified_sources = set()
    for fname in sources:
        if not os.path.isabs(fname):
            fname = os.path.join(src_base_path, fname)
        fname = os.path.abspath(str(fname))
        if fname in hipify_result:
            file_result = hipify_result[fname]
            if file_result['hipified_path'] is not None:
                fname = hipify_result[fname]['hipified_path']
        hipified_sources.add(os.path.relpath(fname, str(src_base_path)))

    if sources_fname is not None:
        with open(sources_fname, "w") as f:
            for fname in hipified_sources:
                f.write(fname + "\n")
        return sources_fname

    return list(hipified_sources)
    

def hipify_sources(te_root: Union[Path, str], src_dir: Union[Path, str],
                   common_headers_dir: Optional[Union[Path, str]],
                   sources: Union[list[Union[Path, str]], Path, str],
                   src_base_path: Union[Path, str]) -> Union[list[str], str]:
    """Hipify source files and return the list of hipified source paths.
    """
    return get_hipified_sources(do_hipify(te_root, src_dir, common_headers_dir),
                                sources, src_base_path)


def copy_hipify_tools(
    src_dir: Union[Path, str],
    dst_dir: Union[Path, str],
) -> None:
    """Copy necessary hipify tools from library root
    src_dir should be the root or Transformer Engine repository.
    """
    if bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        hipify_dir = src_dir / "3rdparty" / "hipify_torch"
        hipify_copy = dst_dir / "3rdparty" / "hipify_torch"
        if hipify_copy.exists():
            shutil.rmtree(hipify_copy)
        shutil.copytree(hipify_dir, hipify_copy)


def clear_hipify_tools_copy(
    dst_dir: Union[Path, str],
) -> None:
    """Clear temporary copies of hipify tools
    """
    hipify_copy = dst_dir / "3rdparty"
    if hipify_copy.exists():
        shutil.rmtree(hipify_copy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hipify TE source files")
    subparsers = parser.add_subparsers(dest="op", help="Operation to perform")

    parser_hipify = subparsers.add_parser("hipify", help="Run hipify on source directory")
    parser_hipify.add_argument("--te-root", type=str, default=str(Path(__file__).parent.parent.parent),
                        help="Root directory of the transformer engine project")
    parser_hipify.add_argument("--src-dir", type=str, required=True,
                        help="Source directory containing CUDA files to be hipified")
    parser_hipify.add_argument("--hipify-result", type=str, required=True,
                        help="JSON file to save hipify results to")

    parser_sources = subparsers.add_parser("get_sources", help="Get hipified sources from hipify results")
    parser_sources.add_argument("--hipify-result", type=str, required=True,
                        help="JSON file containing hipify results")
    parser_sources.add_argument("--sources", type=str, required=True,
                        help="File containing list of source files to be updated with hipified paths")
    parser_sources.add_argument("--base-path", type=str, default=None, dest="src_base_path",
                        help="Base path for computing relative paths of hipified sources")

    args = parser.parse_args()
    if args.op == "hipify":
        print(f"Hipifying sources in {args.src_dir} with TE root {args.te_root}, saving results to {args.hipify_result}")
        do_hipify(args.te_root, args.src_dir, None, args.hipify_result)
    elif args.op == "get_sources":
        print(f"Getting hipified sources from {args.hipify_result} and updating {args.sources} with base path {args.src_base_path}")
        get_hipified_sources(args.hipify_result, args.sources, args.src_base_path)
    else:
        raise ValueError(f"Unsupported operation: {args.op}")
