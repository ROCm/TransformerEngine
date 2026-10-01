#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Decide which CI test suites a pull request needs, from the paths it changes.

Reads changed paths on stdin, one per line. Plain paths and the output of
``git diff --name-status`` are both accepted; a status line contributes every
path on it, so a deletion or a rename still counts against the path it left.

Each path is matched against RULES, first match wins, and the PR runs the union
of what its paths ask for. A path that matches no rule is considered to require all suites.

Only pull requests are gated. For pushes to dev and release branches, for
manual runs, and for a PR labelled ci-skip-scope, the workflow calls this
script with --all, which ignores the rules and selects every suite.

Example:
  git diff --name-status --no-renames dev...HEAD | python3 ci/change_scope.py
"""

import argparse
import re
import sys

# Suite labels, as named in ci/ci_sgpu_queue.conf and ci/ci_mgpu_queue.conf.
# run_queue.sh rejects a --suites name its config does not define, so a label
# renamed there without renaming it here fails loudly instead of gating wrongly.
SGPU_SUITES = ("examples", "torch", "jax", "core")
MGPU_SUITES = ("torch-mgpu", "jax-mgpu")
EVERYTHING = frozenset(SGPU_SUITES + MGPU_SUITES)
NOTHING = frozenset()

TORCH = frozenset({"torch", "torch-mgpu"})
JAX = frozenset({"jax", "jax-mgpu"})

# (pattern, suites, why). First match wins, so a specific pattern must come
# before the general one it carves out of. In a pattern, "**" matches across
# directories and "*" within one; a pattern with no "/" matches only at the
# repository root.
RULES = [
    # Not part of the build or of any test
    ("docs/**", NOTHING, "documentation"),
    ("qa/**", NOTHING, "upstream NVIDIA QA scripts; ROCm CI does not run them"),
    ("ci/README.md", NOTHING, "documentation"),
    ("**/*.md", NOTHING, "documentation"),
    ("*.rst", NOTHING, "documentation"),
    ("LICENSE", NOTHING, "not built or tested"),
    ("CODEOWNERS", NOTHING, "not built or tested"),
    ("Acknowledgements.txt", NOTHING, "not built or tested"),
    ("SECURITY.md", NOTHING, "not built or tested"),
    (".gitignore", NOTHING, "not built or tested"),
    (".clang-format", NOTHING, "lint configuration; the lint workflow covers it"),
    ("pylintrc", NOTHING, "lint configuration; the lint workflow covers it"),
    ("CPPLINT.cfg", NOTHING, "lint configuration; the lint workflow covers it"),
    # Benchmarks: only one of them is run by CI, at TEST_LEVEL 3
    (
        "benchmarks/attention/benchmark_attention_rocm.py",
        frozenset({"torch"}),
        "run by ci/pytorch.sh at level 3",
    ),
    ("benchmarks/**", NOTHING, "not run by CI"),
    # The CI itself
    (".github/workflows/rocm-ci.yml", EVERYTHING, "the CI workflow"),
    (".github/workflows/rocm-ci-dispatch.yml", EVERYTHING, "the CI workflow"),
    (".github/workflows/rocm-wheels-build.yml", EVERYTHING, "the CI build"),
    (".github/scripts/**", EVERYTHING, "CI scripts"),
    (".github/**", NOTHING, "other workflows run on their own triggers"),
    # Test sources. No sGPU test imports from tests/pytorch/distributed, but the
    # distributed tests import tests/pytorch/utils.py; and in JAX the two go
    # both ways (test_fused_attn.py imports distributed_test_base.py, the
    # distributed tests import test_fused_attn, test_permutation and
    # test_fused_router), so only the distributed test files themselves are
    # mGPU-only.
    ("tests/pytorch/distributed/**", frozenset({"torch-mgpu"}), "PyTorch mGPU tests"),
    ("tests/pytorch/attention/*with_cp*", frozenset({"torch-mgpu"}), "PyTorch mGPU tests"),
    ("tests/pytorch/**", TORCH, "PyTorch tests and helpers"),
    ("tests/jax/test_distributed_*.py", frozenset({"jax-mgpu"}), "JAX mGPU tests"),
    ("tests/jax/**", JAX, "JAX tests and helpers"),
    ("tests/cpp/**", frozenset({"core"}), "C++ tests"),
    ("tests/cpp_distributed/**", NOTHING, "not run by ROCm CI"),
    ("examples/pytorch/**", frozenset({"examples"}), "examples suite"),
    ("examples/jax/**", frozenset({"examples"}), "examples suite"),
    ("examples/**", NOTHING, "not run by CI"),
    # Framework bindings. The examples suite runs the PyTorch and JAX MNIST and
    # encoder examples, so it follows either framework.
    ("transformer_engine/pytorch/**", TORCH | {"examples"}, "PyTorch bindings"),
    ("transformer_engine/debug/**", TORCH | {"examples"}, "PyTorch-only debug tools"),
    ("transformer_engine/jax/**", JAX | {"examples"}, "JAX bindings"),
    # Suite scripts
    ("ci/pytorch.sh", TORCH, "PyTorch suite script"),
    ("ci/jax.sh", JAX, "JAX suite script"),
    ("ci/core.sh", frozenset({"core"}), "C++ suite script"),
    # Everything else every suite shares: the C++ library all frameworks link,
    # the build, the submodules and the rest of the CI scripts. The catch-all
    # is spelled out so the table reads as complete.
    ("ci/**", EVERYTHING, "shared CI infrastructure"),
    ("transformer_engine/**", EVERYTHING, "core library, shared by every framework"),
    ("3rdparty/**", EVERYTHING, "submodule, built into the core library"),
    ("build_tools/**", EVERYTHING, "build"),
    ("setup.py", EVERYTHING, "build"),
    ("**", EVERYTHING, "no rule names this path"),
]


def _compile(pattern):
    """A RULES pattern as an anchored regular expression."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


_COMPILED = [(_compile(pattern), pattern, suites, why) for pattern, suites, why in RULES]


def classify(path):
    """The first rule that names ``path``, as ``(pattern, suites, why)``."""
    for regex, pattern, suites, why in _COMPILED:
        if regex.match(path):
            return pattern, suites, why
    raise AssertionError("the catch-all rule matches every path")


def read_paths(lines):
    """Changed paths from plain or ``git diff --name-status`` lines."""
    paths = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        # A status line is "<status>\t<path>[\t<path>]"; a plain path has no tab
        paths.extend(fields[1:] if len(fields) > 1 else fields)
    return paths


def scope(paths):
    """The suites ``paths`` need, and per path the rule that decided it."""
    needed = set()
    decisions = []
    for path in paths:
        pattern, suites, why = classify(path)
        needed |= suites
        decisions.append((path, pattern, suites, why))
    return needed, decisions


def ordered(suites, names):
    """``suites`` restricted to ``names``, in their order, space-separated."""
    return " ".join(name for name in names if name in suites)


def summary_lines(needed, decisions, reason):
    """Markdown for the job summary: the verdict, then why, path by path."""
    sgpu = ordered(needed, SGPU_SUITES) or "none"
    mgpu = ordered(needed, MGPU_SUITES) or "none"
    out = ["## CI scope", ""]
    if reason:
        out += [f"Running every suite: {reason}.", ""]
        return out
    out += [f"- sGPU suites: **{sgpu}**", f"- mGPU suites: **{mgpu}**", ""]
    if not needed:
        out += ["Nothing this PR changes is built or tested by CI, so no test job runs.", ""]
    out += [
        f"<details><summary>Why -- {len(decisions)} changed paths</summary>",
        "",
        "| Path | Rule | Runs | Reason |",
        "|---|---|---|---|",
    ]
    for path, pattern, suites, why in decisions:
        runs = "everything" if suites == EVERYTHING else " ".join(sorted(suites)) or "nothing"
        out.append(f"| `{path}` | `{pattern}` | {runs} | {why} |")
    out += ["", "</details>", ""]
    return out


def main():
    """Read changed paths, print the verdict, and optionally write it for GitHub."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--all",
        metavar="REASON",
        help="skip the rules and select every suite, e.g. for a push or a ci-skip-scope label; "
        "stdin is not read",
    )
    parser.add_argument(
        "--github-output",
        metavar="FILE",
        help="append run_build, sgpu_suites and mgpu_suites to FILE ($GITHUB_OUTPUT)",
    )
    parser.add_argument("--summary", metavar="FILE", help="append a Markdown explanation to FILE")
    args = parser.parse_args()

    if args.all:
        needed, decisions = set(EVERYTHING), []
    else:
        needed, decisions = scope(read_paths(sys.stdin))

    sgpu = ordered(needed, SGPU_SUITES)
    mgpu = ordered(needed, MGPU_SUITES)
    for path, pattern, suites, _ in decisions:
        runs = "everything" if suites == EVERYTHING else " ".join(sorted(suites)) or "nothing"
        print(f"  {path}: {runs}  [{pattern}]", file=sys.stderr)
    print(f"sgpu_suites={sgpu}")
    print(f"mgpu_suites={mgpu}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"run_build={'true' if needed else 'false'}\n")
            handle.write(f"sgpu_suites={sgpu}\n")
            handle.write(f"mgpu_suites={mgpu}\n")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write("\n".join(summary_lines(needed, decisions, args.all)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
