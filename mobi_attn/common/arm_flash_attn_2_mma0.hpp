/**
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */
#pragma once

#include <cstdint>
#include <arm_neon.h>
#include "mobi_attn/common/arm_common.hpp"

namespace mobi_attn {

// =============================================================================
// ASM kernel for all template.
// =============================================================================
// Br=4
// Bc=4
// Can't handle actual br and bc < 4's situation
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_br4_bc4_neon_micro_kernel(
    const float16_t* __restrict__ q_block, const float16_t* __restrict__ k_block,
    float* __restrict__ acc_s, const int32_t dim_size, const int32_t stride_q,
    const int32_t stride_k, const int32_t stride_acc) {
#pragma unroll
  for (int32_t b_r_idx = 0; b_r_idx < 4; ++b_r_idx) {
    const float16_t* q_row = q_block + b_r_idx * stride_q;

    const float16_t* k_row0 = k_block + 0 * stride_k;
    const float16_t* k_row1 = k_block + 1 * stride_k;
    const float16_t* k_row2 = k_block + 2 * stride_k;
    const float16_t* k_row3 = k_block + 3 * stride_k;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    int32_t i = 0;
    for (; i <= dim_size - 8; i += 8) {
      __builtin_prefetch(q_row + i + 64);
      __builtin_prefetch(k_row0 + i + 64);
      __builtin_prefetch(k_row1 + i + 64);

      float16x8_t q_vec = vld1q_f16(q_row + i);

      float16x8_t k_vec0 = vld1q_f16(k_row0 + i);
      float16x8_t k_vec1 = vld1q_f16(k_row1 + i);
      float16x8_t k_vec2 = vld1q_f16(k_row2 + i);
      float16x8_t k_vec3 = vld1q_f16(k_row3 + i);

      sum0 = vfmlalq_low_f16(sum0, q_vec, k_vec0);
      sum0 = vfmlalq_high_f16(sum0, q_vec, k_vec0);

      sum1 = vfmlalq_low_f16(sum1, q_vec, k_vec1);
      sum1 = vfmlalq_high_f16(sum1, q_vec, k_vec1);

      sum2 = vfmlalq_low_f16(sum2, q_vec, k_vec2);
      sum2 = vfmlalq_high_f16(sum2, q_vec, k_vec2);

      sum3 = vfmlalq_low_f16(sum3, q_vec, k_vec3);
      sum3 = vfmlalq_high_f16(sum3, q_vec, k_vec3);
    }

    float total0 = vaddvq_f32(sum0);
    float total1 = vaddvq_f32(sum1);
    float total2 = vaddvq_f32(sum2);
    float total3 = vaddvq_f32(sum3);

    for (; i < dim_size; ++i) {
      const float16_t q_val = q_row[i];
      total0 += q_val * k_row0[i];
      total1 += q_val * k_row1[i];
      total2 += q_val * k_row2[i];
      total3 += q_val * k_row3[i];
    }

    float* acc_s_row = acc_s + b_r_idx * 4;
    acc_s_row[0] = total0;
    acc_s_row[1] = total1;
    acc_s_row[2] = total2;
    acc_s_row[3] = total3;
  }
}

// Br=4
// Bc=4
// For Processing tail, can handle actual br and bc < 4's situation. But acc_s should be always
// tiled into Br(4) x Bc(4)
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_br4x_bc4x_neon_micro_kernel(
    const int32_t Br_n_fixed, const int32_t Bc_n_fixed, const float16_t* __restrict__ q_block,
    const float16_t* __restrict__ k_block, float* __restrict__ acc_s, const int32_t dim_size,
    const int32_t q_stride_size, const int32_t kv_stride_size) {
  for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
    const float16_t* q_block_line = q_block + b_r_idx * q_stride_size;
    for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
      const float16_t* k_block_line = k_block + b_c_idx * kv_stride_size;

      float32x4_t sum0 = vdupq_n_f32(0.0f);
      float32x4_t sum1 = vdupq_n_f32(0.0f);
      float32x4_t sum2 = vdupq_n_f32(0.0f);
      float32x4_t sum3 = vdupq_n_f32(0.0f);

      int i = 0;
      // Main loop
      for (; i <= dim_size - 32; i += 32) {
        // Prefetch data
        __builtin_prefetch(q_block_line + i + 64);
        __builtin_prefetch(k_block_line + i + 64);

        // Load data
        float16x8_t q0 = vld1q_f16(q_block_line + i);
        float16x8_t k0 = vld1q_f16(k_block_line + i);
        float16x8_t q1 = vld1q_f16(q_block_line + i + 8);
        float16x8_t k1 = vld1q_f16(k_block_line + i + 8);
        float16x8_t q2 = vld1q_f16(q_block_line + i + 16);
        float16x8_t k2 = vld1q_f16(k_block_line + i + 16);
        float16x8_t q3 = vld1q_f16(q_block_line + i + 24);
        float16x8_t k3 = vld1q_f16(k_block_line + i + 24);

        // MLA
        sum0 = vfmlalq_high_f16(sum0, q0, k0);
        sum0 = vfmlalq_low_f16(sum0, q0, k0);

        sum1 = vfmlalq_high_f16(sum1, q1, k1);
        sum1 = vfmlalq_low_f16(sum1, q1, k1);

        sum2 = vfmlalq_high_f16(sum2, q2, k2);
        sum2 = vfmlalq_low_f16(sum2, q2, k2);

        sum3 = vfmlalq_high_f16(sum3, q3, k3);
        sum3 = vfmlalq_low_f16(sum3, q3, k3);
      }

      // Reduce
      float total = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

      // Loops left
      for (; i <= dim_size - 8; i += 8) {
        float16x8_t q = vld1q_f16(q_block_line + i);
        float16x8_t k = vld1q_f16(k_block_line + i);
        total += vaddvq_f32(vfmlalq_high_f16(vfmlalq_low_f16(vdupq_n_f32(0), q, k), q, k));
      }

      for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

      // b_r_idx * Bc + b_c_idx, Bc is 4
      acc_s[b_r_idx * 4 + b_c_idx] = total;
    }
  }
}

// Br=1
// Bc=4
// For decode stage. Br=q_seq_len=1, Bc=kv_seq_len
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_br1_bc4_neon_micro_kernel(
    const float16_t* __restrict__ q_block, const float16_t* __restrict__ k_block,
    float* __restrict__ acc_s, const int32_t dim_size, const int32_t stride_q,
    const int32_t stride_k, const int32_t stride_acc) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  const float16_t* q_ptr = q_block;

  const float16_t* k_ptr0 = k_block;
  const float16_t* k_ptr1 = k_block + stride_k;
  const float16_t* k_ptr2 = k_block + 2 * stride_k;
  const float16_t* k_ptr3 = k_block + 3 * stride_k;

  int32_t k = 0;
  for (; k <= dim_size - 16; k += 16) {
    __builtin_prefetch(q_ptr + k + 128);
    __builtin_prefetch(k_ptr0 + k + 128);
    __builtin_prefetch(k_ptr1 + k + 128);
    __builtin_prefetch(k_ptr2 + k + 128);
    __builtin_prefetch(k_ptr3 + k + 128);

    float16x8_t q0 = vld1q_f16(q_ptr + k);
    float16x8_t q1 = vld1q_f16(q_ptr + k + 8);

    float16x8_t k0_0 = vld1q_f16(k_ptr0 + k);
    float16x8_t k0_1 = vld1q_f16(k_ptr0 + k + 8);
    float16x8_t k1_0 = vld1q_f16(k_ptr1 + k);
    float16x8_t k1_1 = vld1q_f16(k_ptr1 + k + 8);
    float16x8_t k2_0 = vld1q_f16(k_ptr2 + k);
    float16x8_t k2_1 = vld1q_f16(k_ptr2 + k + 8);
    float16x8_t k3_0 = vld1q_f16(k_ptr3 + k);
    float16x8_t k3_1 = vld1q_f16(k_ptr3 + k + 8);

    acc0 = vfmlalq_low_f16(acc0, q0, k0_0);
    acc0 = vfmlalq_high_f16(acc0, q0, k0_0);
    acc0 = vfmlalq_low_f16(acc0, q1, k0_1);
    acc0 = vfmlalq_high_f16(acc0, q1, k0_1);

    acc1 = vfmlalq_low_f16(acc1, q0, k1_0);
    acc1 = vfmlalq_high_f16(acc1, q0, k1_0);
    acc1 = vfmlalq_low_f16(acc1, q1, k1_1);
    acc1 = vfmlalq_high_f16(acc1, q1, k1_1);

    acc2 = vfmlalq_low_f16(acc2, q0, k2_0);
    acc2 = vfmlalq_high_f16(acc2, q0, k2_0);
    acc2 = vfmlalq_low_f16(acc2, q1, k2_1);
    acc2 = vfmlalq_high_f16(acc2, q1, k2_1);

    acc3 = vfmlalq_low_f16(acc3, q0, k3_0);
    acc3 = vfmlalq_high_f16(acc3, q0, k3_0);
    acc3 = vfmlalq_low_f16(acc3, q1, k3_1);
    acc3 = vfmlalq_high_f16(acc3, q1, k3_1);
  }

  acc_s[0] = vaddvq_f32(acc0);
  acc_s[1] = vaddvq_f32(acc1);
  acc_s[2] = vaddvq_f32(acc2);
  acc_s[3] = vaddvq_f32(acc3);

  for (; k < dim_size; ++k) {
    const float16_t q_val = q_ptr[k];
    acc_s[0] += (float)q_val * (float)k_ptr0[k];
    acc_s[1] += (float)q_val * (float)k_ptr1[k];
    acc_s[2] += (float)q_val * (float)k_ptr2[k];
    acc_s[3] += (float)q_val * (float)k_ptr3[k];
  }
}

// Br=1
// Bc=4
// For decode stage. Br=q_seq_len=1, Bc=kv_seq_len. For Processing tail, can handle actual bc
// < 4's situation. But acc_s should be always tiled into Br(4) x Bc(4)
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_br1_bc4x_neon_micro_kernel(
    const int32_t Bc_n_fixed, const float16_t* __restrict__ q_block,
    const float16_t* __restrict__ k_block, float* __restrict__ acc_s, const int32_t dim_size,
    const int32_t stride_q, const int32_t stride_k, const int32_t stride_acc) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  const float16_t* q_ptr = q_block;
  const float16_t* k_ptrs[4] = {k_block, k_block + stride_k, k_block + 2 * stride_k,
                                k_block + 3 * stride_k};

  int32_t k = 0;
  for (; k <= dim_size - 16; k += 16) {
    __builtin_prefetch(q_ptr + k + 128);
    for (int i = 0; i < Bc_n_fixed; i++) { __builtin_prefetch(k_ptrs[i] + k + 128); }

    float16x8_t q0 = vld1q_f16(q_ptr + k);
    float16x8_t q1 = vld1q_f16(q_ptr + k + 8);

    switch (Bc_n_fixed) {
      case 4: {
        float16x8_t k3_0 = vld1q_f16(k_ptrs[3] + k);
        float16x8_t k3_1 = vld1q_f16(k_ptrs[3] + k + 8);
        acc3 = vfmlalq_low_f16(acc3, q0, k3_0);
        acc3 = vfmlalq_high_f16(acc3, q0, k3_0);
        acc3 = vfmlalq_low_f16(acc3, q1, k3_1);
        acc3 = vfmlalq_high_f16(acc3, q1, k3_1);
      }
      case 3: {
        float16x8_t k2_0 = vld1q_f16(k_ptrs[2] + k);
        float16x8_t k2_1 = vld1q_f16(k_ptrs[2] + k + 8);
        acc2 = vfmlalq_low_f16(acc2, q0, k2_0);
        acc2 = vfmlalq_high_f16(acc2, q0, k2_0);
        acc2 = vfmlalq_low_f16(acc2, q1, k2_1);
        acc2 = vfmlalq_high_f16(acc2, q1, k2_1);
      }
      case 2: {
        float16x8_t k1_0 = vld1q_f16(k_ptrs[1] + k);
        float16x8_t k1_1 = vld1q_f16(k_ptrs[1] + k + 8);
        acc1 = vfmlalq_low_f16(acc1, q0, k1_0);
        acc1 = vfmlalq_high_f16(acc1, q0, k1_0);
        acc1 = vfmlalq_low_f16(acc1, q1, k1_1);
        acc1 = vfmlalq_high_f16(acc1, q1, k1_1);
      }
      case 1: {
        float16x8_t k0_0 = vld1q_f16(k_ptrs[0] + k);
        float16x8_t k0_1 = vld1q_f16(k_ptrs[0] + k + 8);
        acc0 = vfmlalq_low_f16(acc0, q0, k0_0);
        acc0 = vfmlalq_high_f16(acc0, q0, k0_0);
        acc0 = vfmlalq_low_f16(acc0, q1, k0_1);
        acc0 = vfmlalq_high_f16(acc0, q1, k0_1);
        break;
      }
      default: break;
    }
  }

  switch (Bc_n_fixed) {
    case 4: acc_s[3 * stride_acc] = vaddvq_f32(acc3);
    case 3: acc_s[2 * stride_acc] = vaddvq_f32(acc2);
    case 2: acc_s[1 * stride_acc] = vaddvq_f32(acc1);
    case 1: acc_s[0] = vaddvq_f32(acc0); break;
    default: break;
  }

  for (; k < dim_size; ++k) {
    const float16_t q_val = q_ptr[k];
    switch (Bc_n_fixed) {
      case 4: acc_s[3 * stride_acc] += (float)q_val * (float)k_ptrs[3][k];
      case 3: acc_s[2 * stride_acc] += (float)q_val * (float)k_ptrs[2][k];
      case 2: acc_s[1 * stride_acc] += (float)q_val * (float)k_ptrs[1][k];
      case 1: acc_s[0] += (float)q_val * (float)k_ptrs[0][k]; break;
      default: break;
    }
  }
}

// Br=Br
// Bc=Bc
// For all scenario, but not the most fast version.
template<int Br, int Bc>
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_brx_bcx_neon_micro_kernel(
    const float16_t* __restrict__ q_block, const float16_t* __restrict__ k_block,
    float* __restrict__ acc_s, const int32_t dim_size, const int32_t stride_q,
    const int32_t stride_k, const int32_t stride_acc) {
#pragma unroll
  for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
    const float16_t* q_block_line = q_block + b_r_idx * stride_q;
#pragma unroll
    for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
      const float16_t* k_block_line = k_block + b_c_idx * stride_k;

      float32x4_t sum0 = vdupq_n_f32(0.0f);
      float32x4_t sum1 = vdupq_n_f32(0.0f);
      float32x4_t sum2 = vdupq_n_f32(0.0f);
      float32x4_t sum3 = vdupq_n_f32(0.0f);

      int i = 0;
      // Main loop
      for (; i <= dim_size - 32; i += 32) {
        // Prefetch data
        __builtin_prefetch(q_block_line + i + 64);
        __builtin_prefetch(k_block_line + i + 64);

        // Load data
        float16x8_t q0 = vld1q_f16(q_block_line + i);
        float16x8_t k0 = vld1q_f16(k_block_line + i);
        float16x8_t q1 = vld1q_f16(q_block_line + i + 8);
        float16x8_t k1 = vld1q_f16(k_block_line + i + 8);
        float16x8_t q2 = vld1q_f16(q_block_line + i + 16);
        float16x8_t k2 = vld1q_f16(k_block_line + i + 16);
        float16x8_t q3 = vld1q_f16(q_block_line + i + 24);
        float16x8_t k3 = vld1q_f16(k_block_line + i + 24);

        // MLA
        sum0 = vfmlalq_high_f16(sum0, q0, k0);
        sum0 = vfmlalq_low_f16(sum0, q0, k0);

        sum1 = vfmlalq_high_f16(sum1, q1, k1);
        sum1 = vfmlalq_low_f16(sum1, q1, k1);

        sum2 = vfmlalq_high_f16(sum2, q2, k2);
        sum2 = vfmlalq_low_f16(sum2, q2, k2);

        sum3 = vfmlalq_high_f16(sum3, q3, k3);
        sum3 = vfmlalq_low_f16(sum3, q3, k3);
      }

      // Reduce
      float total = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

      // Loops left
      for (; i <= dim_size - 8; i += 8) {
        float16x8_t q = vld1q_f16(q_block_line + i);
        float16x8_t k = vld1q_f16(k_block_line + i);
        total += vaddvq_f32(vfmlalq_high_f16(vfmlalq_low_f16(vdupq_n_f32(0), q, k), q, k));
      }

      for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

      acc_s[b_r_idx * Bc + b_c_idx] = total;
    }
  }
}

// Br=Br
// Bc=Bc
// For all scenario, but not the most fast version. For processing tail case.
template<int Br, int Bc>
MOBI_ATTN_FORCE_INLINE inline void fa2_mma0_bshd_fp16_brxx_bcxx_neon_micro_kernel(
    const int32_t Br_n_fixed, const int32_t Bc_n_fixed, const float16_t* __restrict__ q_block,
    const float16_t* __restrict__ k_block, float* __restrict__ acc_s, const int32_t dim_size,
    const int32_t stride_q, const int32_t stride_k, const int32_t stride_acc) {
  for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
    const float16_t* q_block_line = q_block + b_r_idx * stride_q;
    for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
      const float16_t* k_block_line = k_block + b_c_idx * stride_k;

      float32x4_t sum0 = vdupq_n_f32(0.0f);
      float32x4_t sum1 = vdupq_n_f32(0.0f);
      float32x4_t sum2 = vdupq_n_f32(0.0f);
      float32x4_t sum3 = vdupq_n_f32(0.0f);

      int i = 0;
      // Main loop
      for (; i <= dim_size - 32; i += 32) {
        // Prefetch data
        __builtin_prefetch(q_block_line + i + 64);
        __builtin_prefetch(k_block_line + i + 64);

        // Load data
        float16x8_t q0 = vld1q_f16(q_block_line + i);
        float16x8_t k0 = vld1q_f16(k_block_line + i);
        float16x8_t q1 = vld1q_f16(q_block_line + i + 8);
        float16x8_t k1 = vld1q_f16(k_block_line + i + 8);
        float16x8_t q2 = vld1q_f16(q_block_line + i + 16);
        float16x8_t k2 = vld1q_f16(k_block_line + i + 16);
        float16x8_t q3 = vld1q_f16(q_block_line + i + 24);
        float16x8_t k3 = vld1q_f16(k_block_line + i + 24);

        // MLA
        sum0 = vfmlalq_high_f16(sum0, q0, k0);
        sum0 = vfmlalq_low_f16(sum0, q0, k0);

        sum1 = vfmlalq_high_f16(sum1, q1, k1);
        sum1 = vfmlalq_low_f16(sum1, q1, k1);

        sum2 = vfmlalq_high_f16(sum2, q2, k2);
        sum2 = vfmlalq_low_f16(sum2, q2, k2);

        sum3 = vfmlalq_high_f16(sum3, q3, k3);
        sum3 = vfmlalq_low_f16(sum3, q3, k3);
      }

      // Reduce
      float total = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

      // Loops left
      for (; i <= dim_size - 8; i += 8) {
        float16x8_t q = vld1q_f16(q_block_line + i);
        float16x8_t k = vld1q_f16(k_block_line + i);
        total += vaddvq_f32(vfmlalq_high_f16(vfmlalq_low_f16(vdupq_n_f32(0), q, k), q, k));
      }

      for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

      acc_s[b_r_idx * Bc + b_c_idx] = total;
    }
  }
}

// =============================================================================
// Traits and Interfaces for user
// =============================================================================
struct Mma0ArmNeonFp16 {
  static constexpr bool kSupportsFp16MMA = true;
};

template<int BlockRow_, int BlockCol_>
struct Mma0BlockShape {
  static constexpr int kBlockRow = BlockRow_;
  static constexpr int kBlockCol = BlockCol_;
};

template<typename ElementA_, typename ElementB_, typename ElementC_>
struct Mma0DataTypes {
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
};

struct Mma0RowMajorLayout {};
struct Mma0ColumnMajorLayout {};

template<typename LayoutA_, typename LayoutB_>
struct Mma0MemoryLayouts {
  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
};

template<typename ArchPolicy_, typename BlockShape_, typename DataTypes_, typename MemoryLayouts_>
struct Mma0Policy {
  using Arch = ArchPolicy_;
  using Shape = BlockShape_;
  using Types = DataTypes_;
  using Layouts = MemoryLayouts_;

  static constexpr int kBlockRow = Shape::kBlockRow;
  static constexpr int kBlockCol = Shape::kBlockCol;
};

template<typename MmaPolicy>
class ArmFa2Mma0;

template<int Br, int Bc>
class ArmFa2Mma0<
    Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<Br, Bc>, Mma0DataTypes<float16_t, float16_t, float>,
               Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>> {
 public:
  using Policy = Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<Br, Bc>,
                            Mma0DataTypes<float16_t, float16_t, float>,
                            Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>;

  using ElementA = typename Policy::Types::ElementA;
  using ElementB = typename Policy::Types::ElementB;
  using ElementC = typename Policy::Types::ElementC;

  static constexpr int kBlockRow = Policy::kBlockRow;
  static constexpr int kBlockCol = Policy::kBlockCol;

  MOBI_ATTN_FORCE_INLINE inline void operator()(const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t stride_q,
                                                const int32_t stride_k,
                                                const int32_t stride_acc) const {
    fa2_mma0_bshd_fp16_brx_bcx_neon_micro_kernel<kBlockRow, kBlockCol>(
        q_block, k_block, acc_s, dim_size, stride_q, stride_k, stride_acc);
  }

  MOBI_ATTN_FORCE_INLINE inline void operator()(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                                const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t q_stride_size,
                                                const int32_t kv_stride_size) const {
    fa2_mma0_bshd_fp16_brxx_bcxx_neon_micro_kernel<kBlockRow, kBlockCol>(
        Br_n_fixed, Bc_n_fixed, q_block, k_block, acc_s, dim_size, q_stride_size, kv_stride_size);
  }
};

template<>
class ArmFa2Mma0<
    Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<4, 4>, Mma0DataTypes<float16_t, float16_t, float>,
               Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>> {
 public:
  using Policy =
      Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<4, 4>, Mma0DataTypes<float16_t, float16_t, float>,
                 Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>;

  using ElementA = typename Policy::Types::ElementA;
  using ElementB = typename Policy::Types::ElementB;
  using ElementC = typename Policy::Types::ElementC;

  static constexpr int kBlockRow = Policy::kBlockRow;
  static constexpr int kBlockCol = Policy::kBlockCol;

  MOBI_ATTN_FORCE_INLINE inline void operator()(const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t stride_q,
                                                const int32_t stride_k,
                                                const int32_t stride_acc) const {
    fa2_mma0_bshd_fp16_br4_bc4_neon_micro_kernel(q_block, k_block, acc_s, dim_size, stride_q,
                                                 stride_k, stride_acc);
  }

  MOBI_ATTN_FORCE_INLINE inline void operator()(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                                const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t q_stride_size,
                                                const int32_t kv_stride_size) const {
    fa2_mma0_bshd_fp16_br4x_bc4x_neon_micro_kernel(Br_n_fixed, Bc_n_fixed, q_block, k_block, acc_s,
                                                   dim_size, q_stride_size, kv_stride_size);
  }
};

template<>
class ArmFa2Mma0<
    Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<1, 4>, Mma0DataTypes<float16_t, float16_t, float>,
               Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>> {
 public:
  using Policy =
      Mma0Policy<Mma0ArmNeonFp16, Mma0BlockShape<1, 4>, Mma0DataTypes<float16_t, float16_t, float>,
                 Mma0MemoryLayouts<Mma0RowMajorLayout, Mma0RowMajorLayout>>;

  using ElementA = typename Policy::Types::ElementA;
  using ElementB = typename Policy::Types::ElementB;
  using ElementC = typename Policy::Types::ElementC;

  static constexpr int kBlockRow = Policy::kBlockRow;
  static constexpr int kBlockCol = Policy::kBlockCol;

  MOBI_ATTN_FORCE_INLINE inline void operator()(const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t stride_q,
                                                const int32_t stride_k,
                                                const int32_t stride_acc) const {
    fa2_mma0_bshd_fp16_br1_bc4_neon_micro_kernel(q_block, k_block, acc_s, dim_size, stride_q,
                                                 stride_k, stride_acc);
  }

  MOBI_ATTN_FORCE_INLINE inline void operator()(const int32_t Bc_n_fixed,
                                                const ElementA* __restrict__ q_block,
                                                const ElementB* __restrict__ k_block,
                                                ElementC* __restrict__ acc_s,
                                                const int32_t dim_size, const int32_t q_stride_size,
                                                const int32_t kv_stride_size) const {
    fa2_mma0_bshd_fp16_br1_bc4x_neon_micro_kernel(Bc_n_fixed, q_block, k_block, acc_s, dim_size,
                                                  q_stride_size, kv_stride_size, 4);
  }
};

}  // namespace mobi_attn
