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
MOBI_ATTN_FORCE_INLINE inline void fa2_mma1_bshd_fp16_br4_bc4_neon_micro_kernel(
    const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
  for (int d_base = 0; d_base < dim_size; d_base += 8) {
    float32x4_t acc0[2], acc1[2], acc2[2], acc3[2];
    if (d_base + 8 < dim_size) {
      __builtin_prefetch(acc_o + d_base + 8, 1, 3);
      __builtin_prefetch(v_block + d_base + 8, 0, 3);
    }

    const int32_t row_stride = dim_size;
    acc0[0] = vld1q_f32(acc_o + 0 * row_stride + d_base);
    acc0[1] = vld1q_f32(acc_o + 0 * row_stride + d_base + 4);
    acc1[0] = vld1q_f32(acc_o + 1 * row_stride + d_base);
    acc1[1] = vld1q_f32(acc_o + 1 * row_stride + d_base + 4);
    acc2[0] = vld1q_f32(acc_o + 2 * row_stride + d_base);
    acc2[1] = vld1q_f32(acc_o + 2 * row_stride + d_base + 4);
    acc3[0] = vld1q_f32(acc_o + 3 * row_stride + d_base);
    acc3[1] = vld1q_f32(acc_o + 3 * row_stride + d_base + 4);

    {
      const float16x8_t v0 = vld1q_f16(v_block + 0 * head_size * dim_size + d_base);
      const float16x8_t w0_vec = vdupq_n_f16(w_block[0 * 4 + 0]);  // w[0][0]
      const float16x8_t w1_vec = vdupq_n_f16(w_block[1 * 4 + 0]);  // w[1][0]
      const float16x8_t w2_vec = vdupq_n_f16(w_block[2 * 4 + 0]);  // w[2][0]
      const float16x8_t w3_vec = vdupq_n_f16(w_block[3 * 4 + 0]);  // w[3][0]

      acc0[0] = vfmlalq_low_f16(acc0[0], v0, w0_vec);
      acc0[1] = vfmlalq_high_f16(acc0[1], v0, w0_vec);
      acc1[0] = vfmlalq_low_f16(acc1[0], v0, w1_vec);
      acc1[1] = vfmlalq_high_f16(acc1[1], v0, w1_vec);
      acc2[0] = vfmlalq_low_f16(acc2[0], v0, w2_vec);
      acc2[1] = vfmlalq_high_f16(acc2[1], v0, w2_vec);
      acc3[0] = vfmlalq_low_f16(acc3[0], v0, w3_vec);
      acc3[1] = vfmlalq_high_f16(acc3[1], v0, w3_vec);
    }

    {
      const float16x8_t v1 = vld1q_f16(v_block + 1 * head_size * dim_size + d_base);
      const float16x8_t w0_vec = vdupq_n_f16(w_block[0 * 4 + 1]);  // w[0][1]
      const float16x8_t w1_vec = vdupq_n_f16(w_block[1 * 4 + 1]);  // w[1][1]
      const float16x8_t w2_vec = vdupq_n_f16(w_block[2 * 4 + 1]);  // w[2][1]
      const float16x8_t w3_vec = vdupq_n_f16(w_block[3 * 4 + 1]);  // w[3][1]

      acc0[0] = vfmlalq_low_f16(acc0[0], v1, w0_vec);
      acc0[1] = vfmlalq_high_f16(acc0[1], v1, w0_vec);
      acc1[0] = vfmlalq_low_f16(acc1[0], v1, w1_vec);
      acc1[1] = vfmlalq_high_f16(acc1[1], v1, w1_vec);
      acc2[0] = vfmlalq_low_f16(acc2[0], v1, w2_vec);
      acc2[1] = vfmlalq_high_f16(acc2[1], v1, w2_vec);
      acc3[0] = vfmlalq_low_f16(acc3[0], v1, w3_vec);
      acc3[1] = vfmlalq_high_f16(acc3[1], v1, w3_vec);
    }

    {
      const float16x8_t v2 = vld1q_f16(v_block + 2 * head_size * dim_size + d_base);
      const float16x8_t w0_vec = vdupq_n_f16(w_block[0 * 4 + 2]);  // w[0][2]
      const float16x8_t w1_vec = vdupq_n_f16(w_block[1 * 4 + 2]);  // w[1][2]
      const float16x8_t w2_vec = vdupq_n_f16(w_block[2 * 4 + 2]);  // w[2][2]
      const float16x8_t w3_vec = vdupq_n_f16(w_block[3 * 4 + 2]);  // w[3][2]

      acc0[0] = vfmlalq_low_f16(acc0[0], v2, w0_vec);
      acc0[1] = vfmlalq_high_f16(acc0[1], v2, w0_vec);
      acc1[0] = vfmlalq_low_f16(acc1[0], v2, w1_vec);
      acc1[1] = vfmlalq_high_f16(acc1[1], v2, w1_vec);
      acc2[0] = vfmlalq_low_f16(acc2[0], v2, w2_vec);
      acc2[1] = vfmlalq_high_f16(acc2[1], v2, w2_vec);
      acc3[0] = vfmlalq_low_f16(acc3[0], v2, w3_vec);
      acc3[1] = vfmlalq_high_f16(acc3[1], v2, w3_vec);
    }

    {
      const float16x8_t v3 = vld1q_f16(v_block + 3 * head_size * dim_size + d_base);
      const float16x8_t w0_vec = vdupq_n_f16(w_block[0 * 4 + 3]);  // w[0][3]
      const float16x8_t w1_vec = vdupq_n_f16(w_block[1 * 4 + 3]);  // w[1][3]
      const float16x8_t w2_vec = vdupq_n_f16(w_block[2 * 4 + 3]);  // w[2][3]
      const float16x8_t w3_vec = vdupq_n_f16(w_block[3 * 4 + 3]);  // w[3][3]

      acc0[0] = vfmlalq_low_f16(acc0[0], v3, w0_vec);
      acc0[1] = vfmlalq_high_f16(acc0[1], v3, w0_vec);
      acc1[0] = vfmlalq_low_f16(acc1[0], v3, w1_vec);
      acc1[1] = vfmlalq_high_f16(acc1[1], v3, w1_vec);
      acc2[0] = vfmlalq_low_f16(acc2[0], v3, w2_vec);
      acc2[1] = vfmlalq_high_f16(acc2[1], v3, w2_vec);
      acc3[0] = vfmlalq_low_f16(acc3[0], v3, w3_vec);
      acc3[1] = vfmlalq_high_f16(acc3[1], v3, w3_vec);
    }

    vst1q_f32(acc_o + 0 * row_stride + d_base, acc0[0]);
    vst1q_f32(acc_o + 0 * row_stride + d_base + 4, acc0[1]);
    vst1q_f32(acc_o + 1 * row_stride + d_base, acc1[0]);
    vst1q_f32(acc_o + 1 * row_stride + d_base + 4, acc1[1]);
    vst1q_f32(acc_o + 2 * row_stride + d_base, acc2[0]);
    vst1q_f32(acc_o + 2 * row_stride + d_base + 4, acc2[1]);
    vst1q_f32(acc_o + 3 * row_stride + d_base, acc3[0]);
    vst1q_f32(acc_o + 3 * row_stride + d_base + 4, acc3[1]);
  }
}

// Br=Br
// Bc=Bc
// Not handle tail case.
template<int Br, int Bc>
MOBI_ATTN_FORCE_INLINE inline void fa2_mma1_bshd_fp16_brx_bcx_neon_micro_kernel(
    const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
#pragma unroll
  for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
    for (int d_base = 0; d_base < dim_size; d_base += 8) {
      float32x4_t acc[2];
      acc[0] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
      acc[1] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base + 4);
#pragma unroll
      for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
        const float16_t w = w_block[b_r_idx * Bc + b_c_idx];
        const float16x8_t w_vec = vdupq_n_f16(w);
        const float16_t* v_ptr = v_block + b_c_idx * head_size * dim_size + d_base;
        const float16x8_t v = vld1q_f16(v_ptr);
        acc[0] = vfmlalq_low_f16(acc[0], w_vec, v);
        acc[1] = vfmlalq_high_f16(acc[1], w_vec, v);
      }
      vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc[0]);
      vst1q_f32(acc_o + b_r_idx * dim_size + d_base + 4, acc[1]);
    }
  }
}

// Br=Br
// Bc=Bc
// Handle tail case.
MOBI_ATTN_FORCE_INLINE inline void fa2_mma1_bshd_fp16_brxx_bcxx_neon_micro_kernel(
    const int32_t Br_n_fixed, const int32_t Bc_n_fixed, const float16_t* __restrict__ w_block,
    const float16_t* __restrict__ v_block, float* __restrict__ acc_o, const int32_t head_size,
    const int32_t dim_size) {
  for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
    for (int d_base = 0; d_base < dim_size; d_base += 8) {
      float32x4_t acc[2];
      acc[0] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
      acc[1] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base + 4);

      for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
        const float16_t w = w_block[b_r_idx * Bc_n_fixed + b_c_idx];
        const float16x8_t w_vec = vdupq_n_f16(w);
        const float16_t* v_ptr = v_block + b_c_idx * head_size * dim_size + d_base;
        const float16x8_t v = vld1q_f16(v_ptr);
        acc[0] = vfmlalq_low_f16(acc[0], w_vec, v);
        acc[1] = vfmlalq_high_f16(acc[1], w_vec, v);
      }
      vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc[0]);
      vst1q_f32(acc_o + b_r_idx * dim_size + d_base + 4, acc[1]);
    }
  }
}

// =============================================================================
// Traits and Interfaces for user
// =============================================================================
struct Mma1ArmNeonFp16 {
  static constexpr bool kSupportsFp16MMA = true;
};

template<int BlockRow_, int BlockCol_>
struct Mma1BlockShape {
  static constexpr int kBlockRow = BlockRow_;
  static constexpr int kBlockCol = BlockCol_;
};

template<typename ElementA_, typename ElementB_, typename ElementC_>
struct Mma1DataTypes {
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
};

struct Mma1RowMajorLayout {};
struct Mma1ColumnMajorLayout {};

template<typename LayoutA_, typename LayoutB_>
struct Mma1MemoryLayouts {
  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
};

template<typename ArchPolicy_, typename BlockShape_, typename DataTypes_, typename MemoryLayouts_>
struct Mma1Policy {
  using Arch = ArchPolicy_;
  using Shape = BlockShape_;
  using Types = DataTypes_;
  using Layouts = MemoryLayouts_;

  static constexpr int kBlockRow = Shape::kBlockRow;
  static constexpr int kBlockCol = Shape::kBlockCol;
};

template<typename Mma1Policy>
class ArmFa2Mma1;

template<>
class ArmFa2Mma1<
    Mma1Policy<Mma1ArmNeonFp16, Mma1BlockShape<4, 4>, Mma1DataTypes<float16_t, float16_t, float>,
               Mma1MemoryLayouts<Mma1RowMajorLayout, Mma1RowMajorLayout>>> {
 public:
  using Policy =
      Mma1Policy<Mma1ArmNeonFp16, Mma1BlockShape<4, 4>, Mma1DataTypes<float16_t, float16_t, float>,
                 Mma1MemoryLayouts<Mma1RowMajorLayout, Mma1RowMajorLayout>>;

  using ElementA = typename Policy::Types::ElementA;
  using ElementB = typename Policy::Types::ElementB;
  using ElementC = typename Policy::Types::ElementC;

  static constexpr int kBlockRow = Policy::kBlockRow;
  static constexpr int kBlockCol = Policy::kBlockCol;

  MOBI_ATTN_FORCE_INLINE inline void operator()(const ElementA* __restrict__ w_block,
                                                const ElementB* __restrict__ v_block,
                                                ElementC* __restrict__ acc_o,
                                                const int32_t head_size,
                                                const int32_t dim_size) const {
    fa2_mma1_bshd_fp16_br4_bc4_neon_micro_kernel(w_block, v_block, acc_o, head_size, dim_size);
  }

  MOBI_ATTN_FORCE_INLINE inline void operator()(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                                const ElementA* __restrict__ w_block,
                                                const ElementB* __restrict__ v_block,
                                                ElementC* __restrict__ acc_o,
                                                const int32_t head_size,
                                                const int32_t dim_size) const {
    fa2_mma1_bshd_fp16_brxx_bcxx_neon_micro_kernel(Br_n_fixed, Bc_n_fixed, w_block, v_block, acc_o,
                                                   head_size, dim_size);
  }
};

template<>
class ArmFa2Mma1<
    Mma1Policy<Mma1ArmNeonFp16, Mma1BlockShape<1, 4>, Mma1DataTypes<float16_t, float16_t, float>,
               Mma1MemoryLayouts<Mma1RowMajorLayout, Mma1RowMajorLayout>>> {
 public:
  using Policy =
      Mma1Policy<Mma1ArmNeonFp16, Mma1BlockShape<1, 4>, Mma1DataTypes<float16_t, float16_t, float>,
                 Mma1MemoryLayouts<Mma1RowMajorLayout, Mma1RowMajorLayout>>;

  using ElementA = typename Policy::Types::ElementA;
  using ElementB = typename Policy::Types::ElementB;
  using ElementC = typename Policy::Types::ElementC;

  static constexpr int kBlockRow = Policy::kBlockRow;
  static constexpr int kBlockCol = Policy::kBlockCol;

  MOBI_ATTN_FORCE_INLINE inline void operator()(const ElementA* __restrict__ w_block,
                                                const ElementB* __restrict__ v_block,
                                                ElementC* __restrict__ acc_o,
                                                const int32_t head_size,
                                                const int32_t dim_size) const {
    fa2_mma1_bshd_fp16_brx_bcx_neon_micro_kernel<1, 4>(w_block, v_block, acc_o, head_size,
                                                       dim_size);
  }

  MOBI_ATTN_FORCE_INLINE inline void operator()(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                                const ElementA* __restrict__ w_block,
                                                const ElementB* __restrict__ v_block,
                                                ElementC* __restrict__ acc_o,
                                                const int32_t head_size,
                                                const int32_t dim_size) const {
    fa2_mma1_bshd_fp16_brxx_bcxx_neon_micro_kernel(Br_n_fixed, Bc_n_fixed, w_block, v_block, acc_o,
                                                   head_size, dim_size);
  }
};

}  // namespace mobi_attn
