/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

 #include <cuda_bf16.h>
 #include <cuda_fp8.h>
 #include <cuda_fp4.h>
 #include <cuda_runtime.h>
 #include <gtest/gtest.h>
 
 #include <transformer_engine/cast.h>
 #include <transformer_engine/activation.h>
 #include "../test_common.h"
 #include "transformer_engine/transformer_engine.h"
 #include <fstream>
 
 using namespace transformer_engine;
 using namespace test;
 
 namespace {
 
 enum ActivationType {
     Identity,
     GeLU,
     SiLU,
     ReLU,
     QGeLU,
     SReLU
 };
 
 double2 cvt_fp4x2_to_double2(fp4e2m1x2 fp4_pair) {
#ifdef __HIP_PLATFORM_AMD__
     uint8_t raw = *reinterpret_cast<uint8_t*>(&fp4_pair);
     // Decode manually
     float lo = E2M1_LUT[raw & 0xF];
     float hi = E2M1_LUT[(raw >> 4) & 0xF];
     return {static_cast<double>(lo), static_cast<double>(hi)};
#else
     const __half2_raw raw_truncated_to_fp4e2m1_pair =
         __nv_cvt_fp4x2_to_halfraw2(*reinterpret_cast<__nv_fp4x2_storage_t*>(&fp4_pair), __NV_E2M1);
 
     const __half2 truncated_to_fp4e2m1_pair(raw_truncated_to_fp4e2m1_pair);
     const double truncated_to_fp4e2m1_x = static_cast<double>(truncated_to_fp4e2m1_pair.x);
     const double truncated_to_fp4e2m1_y = static_cast<double>(truncated_to_fp4e2m1_pair.y);
     return {truncated_to_fp4e2m1_x, truncated_to_fp4e2m1_y};
#endif //#ifndef __HIP_PLATFORM_AMD__
 }
 
 template <typename InputType>
 std::vector<InputType> create_transpose(const InputType* const input, const size_t rows, size_t cols) {
     std::vector<InputType> input_t(cols * rows);
     for (size_t i = 0; i < rows; ++i) {
         for (size_t j = 0; j < cols; ++j) {
             const size_t idx = i * cols + j;
             const size_t idx_t = j * rows + i;
             input_t[idx_t] = input[idx];
         }
     }
     return input_t;
 }
 
// CPU-side mirror of the kernel's compute_e8m0_scale.
// Rounds amax to nearest power of 2, then subtracts FP4 headroom (2).
inline fp8e8m0 compute_e8m0_scale_cpu(float amax, float& scale_reciprocal) {
    if (amax == 0.0f) {
        scale_reciprocal = 1.0f;
        return 127;
    }
    uint32_t amax_bits = *reinterpret_cast<uint32_t*>(&amax);
    amax_bits = (amax_bits + 0x200000u) & 0xFF800000u;
    int exp = ((amax_bits >> 23) & 0xFF) - 127;
    int scale_unbiased = exp - 2;
    scale_unbiased = std::max(-127, std::min(127, scale_unbiased));
    uint32_t rcp_bits = static_cast<uint32_t>(127 - scale_unbiased) << 23;
    scale_reciprocal = *reinterpret_cast<float*>(&rcp_bits);
    return static_cast<fp8e8m0>(scale_unbiased + 127);
}

template <typename InputType>
void quantize_mxfp4_1d(float (*OP)(const float),
                       const InputType* const input,
                       fp4e2m1x2* const output,
                       fp8e8m0* const scales,
                        const size_t rows,
                        const size_t cols,
                        const size_t scales_stride,
                        const float global_amax) {

     constexpr size_t block_size_X = 32;
     const size_t blocks_X = divide_round_up(cols, block_size_X);

     std::array<float, block_size_X> cache_buffer;
     for (size_t i = 0; i < block_size_X; ++i) {
         cache_buffer[i] = 0.0f;
     }

     for (size_t i = 0; i < rows; ++i) {
         for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
             const size_t j_min = block_X * block_size_X;
             const size_t j_max = j_min + block_size_X;

             // Find block amax
             float block_amax = 0.0f;
             for (size_t j = j_min; j < j_max; ++j) {
                 const size_t idx = i * cols + j;
                 const size_t cache_idx = j - j_min;

                 const float input_elt = static_cast<float>(input[idx]);
                 const float act_elt = OP(input_elt);

                 // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                 const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                 cache_buffer[cache_idx] = elt;
                 block_amax = std::max(block_amax, std::abs(elt));
             }

             float scale_reciprocal;
             const fp8e8m0 biased_exponent = compute_e8m0_scale_cpu(block_amax, scale_reciprocal);
             const size_t scale_idx = i * scales_stride + block_X;
             scales[scale_idx] = biased_exponent;
 
             for (size_t j = j_min; j < j_max; j += 2) {
                 const int idx_pair = (i * cols + j) / 2;
                 const int cache_idx_x = j - j_min;
                 const int cache_idx_y = cache_idx_x + 1;
                 const float cached_x = cache_buffer[cache_idx_x];
                 const float cached_y = cache_buffer[cache_idx_y];
                 const float scaled_elt_x = cached_x * scale_reciprocal;
                 const float scaled_elt_y = cached_y * scale_reciprocal;
                 const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};
 
                 fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                 output[idx_pair] = casted_to_e2m1_pair;
 
                 // const double2 truncated_pair = cvt_fp4x2_to_double2(casted_to_e2m1_pair);
             }
         }
     }
 }
 
template <typename InputType>
void compute_ref(float (*OP)(const float),
                 const InputType* input,
                 fp4e2m1x2* output,
                 fp4e2m1x2* output_t,
                 fp8e8m0* scales,
                 fp8e8m0* scales_t,
                 const float global_amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const size_t scales_stride_t)
{
    std::vector<InputType> input_t = create_transpose(input, rows, cols);

    quantize_mxfp4_1d(OP, input, output, scales, rows, cols, scales_stride, global_amax);
    quantize_mxfp4_1d(OP, input_t.data(), output_t, scales_t, cols, rows, scales_stride_t, global_amax);
}
 
 void compare_mxfp4_tensors(const std::string& name,
                            const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                            const int rows, const int cols,
                            double atol = 1e-5, double rtol = 1e-8) {
     std::vector<std::string> mismatch_messages;
     size_t total_mismatches = 0;
 
     for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; j += 2) {
             const int idx = i * cols + j;
             double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
             double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));
 
             for (int k = 0; k < 2; ++k) {
                 const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                 const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);
 
                 bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
                 /* For Float32 the floating point comparison is enough to error out */
                 bool assertion = false;
                 if (mismatch && !assertion) {
                     /* Check if it is just a failure of round to nearest choosing different
                         side of the real value */
                     const double mean = (t + r) / 2;
                     const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
                     const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
                     const double cast_mean_p = static_cast<double>(static_cast<fp4e2m1>(mean_p));
                     const double cast_mean_m = static_cast<double>(static_cast<fp4e2m1>(mean_m));
                     assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
                 }
                 if (assertion) {
                     total_mismatches++;
                     std::string msg = "Mismatch at place (" + std::to_string(idx + k) + "): " +
                                     std::to_string(t) + " vs " + std::to_string(r) +
                                     " (abs_diff: " + std::to_string(fabs(t - r)) +
                                     ", rel_diff: " + std::to_string(r == 0 ? 0.0 : fabs((t - r) / r)) + ")";
                     mismatch_messages.push_back(msg);
 
                     // Optional: limit number of detailed messages to avoid overwhelming output
                     if (mismatch_messages.size() <= 100) {
                         std::cout << "Error in tensor " << name << ": " << msg << std::endl;
                     }
                 }
             }
         }
     }
 
     // Always report summary - either success or failure
     std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
     std::cout << "Total elements checked: " << (rows * cols) << std::endl;
 
     if (total_mismatches > 0) {
         std::cout << "STATUS: FAILED for output" << std::endl;
         std::cout << "Total mismatches found: " << total_mismatches << std::endl;
         std::cout << "Mismatch rate: " << (100.0 * total_mismatches) / (rows * cols) << "%" << std::endl;
         if (mismatch_messages.size() > 100) {
             std::cout << "... and " << (mismatch_messages.size() - 100) << " more mismatches (showing first 100)" << std::endl;
         }
         std::cout << "============================" << std::endl;
 
         GTEST_FAIL() << "Found " << total_mismatches << " mismatches in tensor " << name;
     } else {
         std::cout << "STATUS: PASSED for output" << std::endl;
         std::cout << "All elements match within tolerance!" << std::endl;
         std::cout << "Tensor " << name << " is IDENTICAL to reference" << std::endl;
         std::cout << "============================" << std::endl;
     }
 }
 
 // Optional: Function to dump tensor data to files for detailed analysis
 void dump_mxfp4_tensor_data(const std::string& prefix,
                             const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                             const int rows, const int cols) {
     std::string test_file = prefix + "_test.txt";
     std::string ref_file = prefix + "_ref.txt";
     std::string diff_file = prefix + "_diff.txt";
 
     std::ofstream test_out(test_file);
     std::ofstream ref_out(ref_file);
     std::ofstream diff_out(diff_file);
 
     if (test_out.is_open() && ref_out.is_open() && diff_out.is_open()) {
         for (int i = 0; i < rows; ++i) {
             for (int j = 0; j < cols; j += 2) {
                 const int idx = i * cols + j;
                 double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
                 double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));
 
                 for (int k = 0; k < 2; ++k) {
                     const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                     const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);
                     const int pos = idx + k;
 
                     test_out << "pos[" << pos << "] = " << t << std::endl;
                     ref_out << "pos[" << pos << "] = " << r << std::endl;
                     diff_out << "pos[" << pos << "] test=" << t << " ref=" << r
                             << " abs_diff=" << fabs(t - r)
                             << " rel_diff=" << (r == 0 ? 0.0 : fabs((t - r) / r)) << std::endl;
                 }
             }
         }
         std::cout << "DEBUG: Dumped tensor data to files: " << test_file << ", " << ref_file << ", " << diff_file << std::endl;
     } else {
         std::cout << "WARNING: Could not open files for tensor data dump" << std::endl;
     }
 }
 
 void print_detailed_tensor_comparison(const std::string& name,
                                      const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                                      const int rows, const int cols) {
     printf("\n=== DETAILED COMPARISON for %s (%d×%d = %d elements) ===\n",
            name.c_str(), rows, cols, rows * cols);
 
     const int total_elements = rows * cols;
     const int check_count = 128;
 
     printf("--- FIRST %d ELEMENTS ---\n", check_count);
     printf("Index | Test_Value    | Ref_Value     | Match\n");
     printf("------|---------------|---------------|-------\n");
     for (int i = 0; i < std::min(check_count, total_elements); ++i) {
         double2 test_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[i/2]));
         double2 ref_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[i/2]));
 
         double t = (i % 2 == 0) ? test_pair.x : test_pair.y;
         double r = (i % 2 == 0) ? ref_pair.x : ref_pair.y;
         bool match = (fabs(t - r) < 1e-6);
 
         printf("%5d | %13.6f | %13.6f | %s\n", i, t, r, match ? "✓" : "✗");
     }
 
     if (total_elements > 2 * check_count) {
         printf("\n--- LAST %d ELEMENTS ---\n", check_count);
         printf("Index | Test_Value    | Ref_Value     | Match\n");
         printf("------|---------------|---------------|-------\n");
         for (int i = total_elements - check_count; i < total_elements; ++i) {
             double2 test_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[i/2]));
             double2 ref_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[i/2]));
 
             double t = (i % 2 == 0) ? test_pair.x : test_pair.y;
             double r = (i % 2 == 0) ? ref_pair.x : ref_pair.y;
             bool match = (fabs(t - r) < 1e-6);
 
             printf("%5d | %13.6f | %13.6f | %s\n", i, t, r, match ? "✓" : "✗");
         }
     }
     printf("==================================\n");
 }
 
 void compareResults_mxfp4(const Tensor &test,
                           const void *ref, const void *ref_t, const int rows, const int cols,
                           double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true, bool dump_data = false) {
     if (if_on_gpus) test.to_cpu();
 
     const fp4e2m1 *test_data = test.rowwise_cpu_dptr<fp4e2m1>();
     const fp4e2m1 *test_data_t = test.columnwise_cpu_dptr<fp4e2m1>();
     const fp4e2m1 *ref_data = reinterpret_cast<const fp4e2m1*>(ref);
     const fp4e2m1 *ref_data_t = reinterpret_cast<const fp4e2m1*>(ref_t);
 
     // Print detailed element-by-element comparison
     // print_detailed_tensor_comparison("output", test_data, ref_data, rows, cols);
     // print_detailed_tensor_comparison("output_t", test_data_t, ref_data_t, cols, rows);
 
     // Optionally dump tensor data to files for detailed analysis
     if (dump_data) {
         dump_mxfp4_tensor_data("output", test_data, ref_data, rows, cols);
         dump_mxfp4_tensor_data("output_t", test_data_t, ref_data_t, cols, rows);
     }
 
     compare_mxfp4_tensors("output", test_data, ref_data, rows, cols, atol, rtol);
     compare_mxfp4_tensors("output_t", test_data_t, ref_data_t, cols, rows, atol, rtol);
 }
 
 template <typename InputType>
 void performTest(float (*OP)(const float),
                  const std::vector<size_t>& shape) {
     using namespace test;
 
     DType itype = TypeInfo<InputType>::dtype;
     DType otype = DType::kFloat4E2M1;
 
     const size_t rows = first_dimension(shape);
     const size_t cols = last_dimension(shape);
 
    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 32, NVTE_MXFP4_1D_SCALING);
    const std::array<size_t,4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 32, NVTE_MXFP4_1D_SCALING);
 
     const size_t unpadded_blocks_Y = scale_dims[0];
     const size_t unpadded_blocks_X = scale_dims[1];
     const size_t blocks_Y = scale_dims[2];
     const size_t blocks_X = scale_dims[3];
     const size_t scales_stride = blocks_X;
 
     const size_t unpadded_blocks_Y_t = scale_dims_t[0];
     const size_t unpadded_blocks_X_t = scale_dims_t[1];
     const size_t blocks_Y_t = scale_dims_t[2];
     const size_t blocks_X_t = scale_dims_t[3];
     const size_t scales_stride_t = blocks_X_t;
 
     Tensor input("input", shape, itype);
     Tensor output("output", shape, otype, true, true, NVTE_MXFP4_1D_SCALING);
 
     std::unique_ptr<fp4e2m1x2[]> ref_output   = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
     std::unique_ptr<fp4e2m1x2[]> ref_output_t = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
     std::unique_ptr<fp8e8m0[]> ref_scales     = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);
     std::unique_ptr<fp8e8m0[]> ref_scales_t   = std::make_unique<fp8e8m0[]>(blocks_Y_t * blocks_X_t);
 
     fillCase<fp32>(&input, InputsFillCase::uniform);

     compute_ref<InputType>(OP,
                            input.rowwise_cpu_dptr<InputType>(),
                            ref_output.get(),
                            ref_output_t.get(),
                            ref_scales.get(),
                            ref_scales_t.get(),
                            output.scale(),
                            rows,
                            cols,
                            scales_stride,
                            scales_stride_t);

     QuantizationConfigWrapper quant_config;

     if (OP == &gelu) {
         nvte_gelu(input.data(), output.data(), 0);
     } else if (OP == &silu) {
         nvte_silu(input.data(), output.data(), 0);
     } else if (OP == &relu) {
         nvte_relu(input.data(), output.data(), 0);
     } else if (OP == &qgelu) {
         nvte_qgelu(input.data(), output.data(), 0);
     } else if (OP == &srelu) {
         nvte_srelu(input.data(), output.data(), 0);
     } else {
         nvte_quantize_v2(input.data(), output.data(), quant_config, 0);
     }

     cudaDeviceSynchronize();
     auto err = cudaGetLastError();
     ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

     const double atol = 0.05;
     const double rtol = 0.1;

     compareResults_mxfp4(output, ref_output.get(), ref_output_t.get(), rows, cols, atol, rtol, true, false);

     size_t scale_mismatches_num = 0;
#ifdef USE_ROCM
     std::vector<size_t> mismatches_scales_indices;
#endif

     compare_scaling_factors<fp8e8m0>("scales", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                       ref_scales.get(),
                                       unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
#ifdef USE_ROCM
                                       mismatches_scales_indices,
#endif
                                       scale_mismatches_num);

     compare_scaling_factors<fp8e8m0>("scales_t", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                       ref_scales_t.get(),
                                       unpadded_blocks_Y_t, unpadded_blocks_X_t, scales_stride_t,
#ifdef USE_ROCM
                                       mismatches_scales_indices,
#endif
                                       scale_mismatches_num);
}
 
 std::vector<std::vector<size_t>> tensor_dims = {
     {32, 32},
     {32, 64},
     {64, 32},
     {64, 96},
     {128, 128},
     {256, 256},
     {512, 512},
     {1024, 1024},
     {2048, 2048},
     {128, 256},
     {8192, 128},
     {2048, 160},
     {8, 32, 1024},
     {16, 8, 4, 512},
     {1024, 16384},
     {4096, 13312},
 };
 
 // Only the Identity activation is currently supported.
 std::vector<ActivationType> Activation_types = {
     ActivationType::Identity
 };
 
 }  // namespace
 
 class FusedCastTransposeMXFP4TestSuite : public ::testing::TestWithParam
     <std::tuple<ActivationType,
                 std::vector<size_t>,
                 transformer_engine::DType>> {};
 
 TEST_P(FusedCastTransposeMXFP4TestSuite, TestFusedCastTransposeMXFP4) {
#ifdef __HIP_PLATFORM_AMD__
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    if (!(prop.major > 9 || (prop.major == 9 && prop.minor == 5))) {
        GTEST_SKIP() << "MXFP4 requires gfx950";
    }
#else
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
#endif
 
     using namespace transformer_engine;
     using namespace test;
 
     const ActivationType Act_type = std::get<0>(GetParam());
     const auto tensor_dims = std::get<1>(GetParam());
     const DType input_type = std::get<2>(GetParam());
 
     // Skip tests if the input tensor is 1D
     if (tensor_dims.size() < 2) {
         GTEST_SKIP();
     }
 
     // Forward activations
     auto OP = &identity;
     switch (Act_type) {
         case ActivationType::GeLU: OP = &gelu; break;
         case ActivationType::SiLU: OP = &silu; break;
         case ActivationType::ReLU: OP = &relu; break;
         case ActivationType::QGeLU: OP = &qgelu; break;
         case ActivationType::SReLU: OP = &srelu; break;
     }
 
     TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
         performTest<InputType>(OP, tensor_dims);
     );
 }
 
 std::string to_string(const ActivationType Act_type) {
     switch (Act_type) {
         case ActivationType::Identity:  return "CAST_ONLY";
         case ActivationType::GeLU:      return "GeLU";
         case ActivationType::SiLU:      return "SiLU";
         case ActivationType::ReLU:      return "ReLU";
         case ActivationType::QGeLU:     return "QGeLU";
         case ActivationType::SReLU:     return "SReLU";
         default: return "";
     }
 }
 
 INSTANTIATE_TEST_SUITE_P(
     OperatorTest,
     FusedCastTransposeMXFP4TestSuite,
     ::testing::Combine(
         ::testing::ValuesIn(Activation_types),
         ::testing::ValuesIn(tensor_dims),
         ::testing::Values(DType::kBFloat16)),
     [](const testing::TestParamInfo<FusedCastTransposeMXFP4TestSuite::ParamType>& info) {
         std::string name = to_string(std::get<0>(info.param));
       const auto& shape = std::get<1>(info.param);
       for ( const auto& s: shape) {
         name += "X" + std::to_string(s);
       }
       name += "X" + test::typeName(std::get<2>(info.param));
         return name;
     });
