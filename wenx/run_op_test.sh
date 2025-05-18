
# mkdir -p build && cd build
# cmake ..
# make -j

# ./build/test_cast_transpose

echo "====================="
echo "cuda kernel"
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
# ./operator/test_operator
./operator/test_operator --gtest_filter=OperatorTest/CTTestSuite.TestCastTranspose/bfloat16Xfloat8e5m2X32768X57344
# ./operator/test_operator --gtest_filter=OperatorTest/CTTestSuite.TestCastTranspose/bfloat16Xfloat8e5m2X32768X8192
# ./operator/test_operator --gtest_list_tests

echo "====================="
echo "triton kernel"
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
# ./operator/test_operator
