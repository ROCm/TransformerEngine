#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# Run small-shape tests on gfx1250 (B0 stepping) emulated on the CPU by rocjitsu
# (https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu), so no GPU is needed.
# `ci/run_queue.sh --queue emu` runs the items side by side; run directly, this script runs
# them one after another.

# CI installs only TE's PyTorch extension, although the image also has JAX
export NVTE_FRAMEWORK=pytorch

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

# rocm-systems commit to build rocjitsu from; bump deliberately.
: ${ROCJITSU_REF:=2673851a7dcc2ad76827fcc01b94412912fe7c8c}
: ${ROCJITSU_HOME:=${TE_PATH}build/rocjitsu}
: ${ROCJITSU_CPP_BUILD_DIR:=${TE_PATH}build/rocjitsu_cpp_tests}
# Emulator threads per test process, gtest shards per C++ test group, timeout per process in seconds
: ${ROCJITSU_CPU_BUDGET:=4}
: ${ROCJITSU_SHARDS:=4}
: ${ROCJITSU_TIMEOUT:=3600}

ROCJITSU=$ROCJITSU_HOME/build/tools/rocjitsu/rocjitsu
B0_CONFIG=$ROCJITSU_HOME/gfx1250_mi455x.json
TEST_OPERATOR=$ROCJITSU_CPP_BUILD_DIR/operator/test_operator

# Shapes that take minutes per test case under emulation, as they appear in gtest names.
LARGE_SHAPES="*8192X7168*:*2048X12288*:*256X65536*:*65536X*:*4096X13312*:*1024X16384*:*X768X6144X*"
LARGE_SHAPES="$LARGE_SHAPES:*X16384X1616*:*32X1481X677*:*2304x768x4096*:*768x3072x4096*:*768x768x4096*"
# Known gfx1250 MXFP8 bugs (2D scaling applied as 1D, compact scales in GEMM-swizzled outputs); delete once fixed
KNOWN_FAILURES="*/CastMXFP82DQuantizationTestSuite.*:*/SwizzledScalesFusedCastMXFP8TestSuite.*"

install_prerequisites() {
    # The CI image carries no gfx1250 device code for the ROCm libraries and PyTorch.
    _pkgs=`python3 - <<'EOF'
from importlib.metadata import PackageNotFoundError, version

for dist, device in (("rocm-sdk-core", "rocm-sdk-device-gfx1250"),
                     ("torch", "amd-torch-device-gfx1250"),
                     ("torchvision", "amd-torchvision-device-gfx1250")):
    try:
        print(f"{device}=={version(dist)}")
    except PackageNotFoundError:
        pass
EOF`
    pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next pytest-timeout $_pkgs
    rc=$?
    if [ $rc -ne 0 ]; then
        script_error "Failed to install test prerequisites"
        exit $rc
    fi
}

build_rocjitsu() {
    if [ -x "$ROCJITSU" -a -f "$B0_CONFIG" ] && [ "`cat $ROCJITSU_HOME/ref 2>/dev/null`" = "$ROCJITSU_REF" ]; then
        echo "Reusing rocjitsu $ROCJITSU_REF from $ROCJITSU_HOME"
        return 0
    fi
    echo "Building rocjitsu $ROCJITSU_REF in $ROCJITSU_HOME"
    _src=$ROCJITSU_HOME/src
    rm -rf "$_src" "$ROCJITSU_HOME/build" "$ROCJITSU_HOME/ref"
    mkdir -p "$ROCJITSU_HOME"
    (
        git init -q "$_src" &&
        git -C "$_src" remote add origin https://github.com/ROCm/rocm-systems.git &&
        git -C "$_src" sparse-checkout set emulation/rocjitsu runtimes/api-headers &&
        git -C "$_src" fetch -q --depth 1 --filter=blob:none origin "$ROCJITSU_REF" &&
        git -C "$_src" checkout -q FETCH_HEAD &&
        cmake -S "$_src/emulation/rocjitsu" -B "$ROCJITSU_HOME/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF &&
        cmake --build "$ROCJITSU_HOME/build" --target rocjitsu_bin rocjitsu_shared &&
        cp "$_src/emulation/rocjitsu/configs/gfx1250_mi455x.json" "$B0_CONFIG"
    ) > "$ROCJITSU_HOME/build.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        script_error "Failed to build rocjitsu"
        cat "$ROCJITSU_HOME/build.log"
        return $rc
    fi
    # Not needed to run, and its root-only git files break the CI cache
    rm -rf "$_src"
    echo "$ROCJITSU_REF" > "$ROCJITSU_HOME/ref"
}

build_cpp_tests() {
    mkdir -p "$ROCJITSU_CPP_BUILD_DIR"
    _log=$ROCJITSU_CPP_BUILD_DIR/te_ci_build.log
    (cd ${TE_PATH}tests/cpp &&
        NVTE_ROCM_ARCH=gfx1250 cmake -GNinja -B"$ROCJITSU_CPP_BUILD_DIR" . &&
        NVTE_ROCM_ARCH=gfx1250 cmake --build "$ROCJITSU_CPP_BUILD_DIR"
    ) > "$_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        script_error "Failed to build cpp test suite"
        cat "$_log"
    fi
    return $rc
}

# A failed check that the test's own JUnit XML cannot show. args: file_tag test_name message
write_failure_junitxml() {
    test -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" || return 0
    cat > "$JUNITXML_PREFIX$1$JUNITXML_SUFFIX" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="rocjitsu" errors="0" failures="1" skipped="0" tests="1">
<testcase classname="$1" name="$2"><failure message="$3"></failure></testcase>
</testsuite></testsuites>
EOF
}

# Run one shard of a C++ test group on the emulated GPU. args: tag shard gtest_filter
gtest_run() {
    local tag=$1 junitxml="" out rc start_ts
    start_ts=`date +%s`
    echo "Run [$tag] at `time_elapsed $TEST_START_TS`"
    if [ -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" ]; then
        junitxml="--gtest_output=xml:${JUNITXML_PREFIX}${tag}${JUNITXML_SUFFIX}"
        rm -f "${JUNITXML_PREFIX}${tag}"*"${JUNITXML_SUFFIX}"
    fi
    out=`mktemp`
    GTEST_SHARD_INDEX=$2 GTEST_TOTAL_SHARDS=$ROCJITSU_SHARDS \
        timeout -k 60 $ROCJITSU_TIMEOUT "$ROCJITSU" --config "$B0_CONFIG" --cpu-thread-budget $ROCJITSU_CPU_BUDGET -- \
        "$TEST_OPERATOR" --gtest_filter="$3" --gtest_brief=1 --gtest_color=no $junitxml 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        test_run_error "[$tag]"
        if [ -n "$junitxml" -a ! -s "${JUNITXML_PREFIX}${tag}${JUNITXML_SUFFIX}" ]; then
            if [ $rc -eq 124 -o $rc -eq 137 ]; then
                write_failure_junitxml $tag "(no JUnit XML written)" "test_operator timed out after ${ROCJITSU_TIMEOUT}s"
            else
                write_failure_junitxml $tag "(no JUnit XML written)" "test_operator exited $rc without writing JUnit XML"
            fi
        fi
    fi
    local waits=`grep -c "memory-wait" "$out"`
    test "${waits:-0}" -gt 0 && echo "rocjitsu reported $waits memory-wait diagnostic(s) in [$tag]"
    rm -f "$out"
    echo "Done [$tag] in `time_elapsed $start_ts`"
}

# A C++ test group, split into ROCJITSU_SHARDS items. args: level name gtest_filter
cpp() {
    check_level $1 || return
    local shard tag
    for shard in `seq 0 $((ROCJITSU_SHARDS - 1))`; do
        tag=cpp_$2.b0.$shard
        check_test_filter $tag || continue
        if [ -n "$TE_CI_LIST_ITEMS" ]; then
            echo "TE_CI_ITEM $tag"
        else
            gtest_run $tag $shard "$3-$LARGE_SHAPES:$KNOWN_FAILURES"
        fi
    done
}

# A pytest file on the emulated GPU. args: level test_file [pytest args...]
py() {
    TEST_LAUNCHER="timeout -k 60 $ROCJITSU_TIMEOUT $ROCJITSU --config $B0_CONFIG --cpu-thread-budget $ROCJITSU_CPU_BUDGET --" \
        pytest_run b0 "" "$@"
}

run_test_config() {
    MXFP8="OperatorTest_FusedCastMXFP8*:OperatorTest_CastMXFP8_2DQuantization*:OperatorTest/DequantizeMXFP8*"
    MXFP8="$MXFP8:OperatorTest/CastMXFP8_GatedActTestSuite.*:OperatorTest/MultiQuantizeMXFP8TestSuite.*"
    MXFP8="$MXFP8:OperatorTest/MxNormTestSuite.*:OperatorTest/MxSwizzleTestSuite.*"
    FP8="OperatorTest/CTTestSuite.*:OperatorTest/CastTestSuite.*:OperatorTest/CTDBias*:OperatorTest/CastDBias*"
    FP8="$FP8:OperatorTest/CTCSTestSuite.*:OperatorTest/CastCSTestSuite.*:OperatorTest/MultiCastTranspose*"

    cpp 1 mxfp8 "$MXFP8"
    cpp 1 nvfp4 "*NVFP4*"
    cpp 1 mxfp4 "*MXFP4*"
    cpp 1 fp8 "$FP8"
    cpp 1 norm "OperatorTest/NormTestSuite.*"
    cpp 1 gemm "OperatorTest/GEMMTestSuite.*:OperatorTestFP8/FP8GEMMTestSuite.*"
    py 1 nvfp4/test_nvfp4_quantize_exact.py -k "not 1024 and not 2048 and not 8192"
}

start_message
if [ -z "$TEST_SGPU" ]; then
    return_run_results
    exit 0
fi

# ci/run_queue.sh hands out CPU slots as device ids; every emulated process has one GPU of its own.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# Keep ROCr from loading its gfx1250 B0-to-A0 hotswap hook (opt-out in older releases such as the image's).
export HSA_HOTSWAP_DISABLE=1 HSA_HOTSWAP_ENABLE=0
# The tests' CPU reference code (OpenMP, PyTorch) would otherwise start a thread per CPU in every item.
: ${OMP_NUM_THREADS:=$ROCJITSU_CPU_BUDGET}
export OMP_NUM_THREADS

#A scheduler runs this setup once with TE_CI_SETUP_ONLY, then skips it for each item (TE_CI_SKIP_SETUP)
if check_setup_needed; then
    install_prerequisites
    build_rocjitsu || exit $?
    build_cpp_tests || exit $?
    test -n "$TE_CI_SETUP_ONLY" && exit 0
fi

run_test_config

return_run_results
