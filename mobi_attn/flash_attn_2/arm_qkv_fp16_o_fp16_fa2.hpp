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

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#error AArch64, FEAT_FP16 is needed to compile this file. Pls trying to recompile with options: -march=armv8.2-a+fp16+dotprod;-ffast-math
#else

#include <omp.h>
#include <cstdint>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include <arm_neon.h>
#include "mobi_attn/utils/arm_math.hpp"

namespace mobi_attn {

#define NEG_INF std::numeric_limits<float>::lowest()

// QKV: FP16, BSHD
// O: FP16, BSHD
// ACC: FP32
template<int32_t Br, int32_t Bc, int32_t threads = 2, bool HP = false>
struct NEON_FA_2_MHA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL {
  using dtype_t = float16_t;
  using acc_dtype_t = float32_t;

  void init_workspace(dtype_t* acc_s_cast, acc_dtype_t* acc_o, acc_dtype_t* acc_s,
                      acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev,
                      acc_dtype_t* score_scale, acc_dtype_t* score_sum) {
    acc_s_cast_ = acc_s_cast;        // [threads][Br * Bc]
    acc_o_ = acc_o;                  // [threads][Br * dim_size]
    acc_s_ = acc_s;                  // [threads][Br * Bc]
    logsum_ = logsum;                // [threads][Br]
    scoremax_ = scoremax;            // [threads][Br]
    scoremax_prev_ = scoremax_prev;  // [threads][Br]
    score_scale_ = score_scale;      // [threads][Br]
    score_sum_ = score_sum;          // [threads][Br]
  }

  // For prefill case.
  inline void __fa2_prefill_append(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
                                   const dtype_t* __restrict__ V, dtype_t* __restrict__ O,
                                   const int32_t batch_size, const int32_t head_size,
                                   const int32_t seq_size_q, const int32_t seq_size_k,
                                   const int32_t dim_size, bool causal_mask = true) {
    // Constant
    const int32_t Tr = seq_size_q / Br;
    const int32_t Tr_left = seq_size_q % Br;
    const int32_t Tc = seq_size_k / Bc;
    const int32_t Tc_left = seq_size_k % Bc;

    scale_ = sqrt(1.0 / dim_size);

    // Loops
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME: Make dynamic threads dispatching available in head_size level.
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
      for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
        const int32_t thread_id = omp_get_thread_num();
        const int32_t this_thread_head = h_idx;

        // Loop S_Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                    acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * Br * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, t_r_idx, t_c_idx,
                 seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax(acc_s_ + thread_id * Br * Bc, acc_s_cast_ + thread_id * Br * Bc,
                    scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br,
                    score_scale_ + thread_id * Br, score_sum_ + thread_id * Br,
                    logsum_ + thread_id * Br, t_r_idx, t_c_idx, seq_size_q, seq_size_k,
                    causal_mask);

            // Rescale
            rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q,
                    seq_size_k, causal_mask);

            // W @ V
            mma1(acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, head_size, dim_size, t_r_idx,
                 t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          // Process the last block of KV
          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * Br * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_pa_n_fixed(Br, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size,
                            t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_pa_n_fixed(Br, Tc_left, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, t_r_idx, Tc,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Br, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx,
                               Tc, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Br, Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o,
                            head_size, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
                              O + b_idx * seq_size_q * head_size * dim_size
                                  + t_r_idx * Br * head_size * dim_size
                                  + this_thread_head * dim_size,
                              t_r_idx, head_size, dim_size);
        }

        // Process left Q
        if (Tr_left) {
          // Init all temps
          init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                    acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0_pa_n_fixed(Tr_left, Bc, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size,
                            Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_pa_n_fixed(Tr_left, Bc, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, Tr, t_c_idx,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Tr_left, Bc, acc_o, score_scale_ + thread_id * Br, dim_size, Tr,
                               t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Tr_left, Bc, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o,
                            head_size, dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          // Process the last block of KV
          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_pa_n_fixed(Tr_left, Tc_left, tile_q, tile_k, tile_acc_s, dim_size,
                            head_size * dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_pa_n_fixed(Tr_left, Tc_left, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, Tr, Tc,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Tr_left, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, Tr,
                               Tc, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Tr_left, Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o,
                            head_size, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_pa_n_fixed(
              Tr_left, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
              O + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size
                  + this_thread_head * dim_size,
              Tr, head_size, dim_size);
        }
      }
    }
  }

  // For decode case.
  inline void __fa2_decode(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
                           const dtype_t* __restrict__ V, dtype_t* __restrict__ O,
                           const int32_t batch_size, const int32_t head_size,
                           const int32_t seq_size_q, const int32_t seq_size_k,
                           const int32_t dim_size, bool causal_mask = true) {
    // Constant
    const int32_t Tr = 1;
    const int32_t Tc = seq_size_k / Bc;
    const int32_t Tc_left = seq_size_k % Bc;
    // scale_ = sqrt(1.0 / dim_size) * 1.44269504;
    scale_ = sqrt(1.0 / dim_size);

    // Loops
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME: Make dynamic threads dispatching available in head_size level.
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
      for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
        const int32_t thread_id = omp_get_thread_num();
        const int32_t this_thread_head = h_idx;

        // Loop Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp_d(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                      acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * 1 * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + t_c_idx * Bc * head_size * dim_size
                                    + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * 1 * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0_d(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, t_r_idx, t_c_idx,
                   seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_d(acc_s_ + thread_id * Br * Bc, acc_s_cast_ + thread_id * Br * Bc,
                      scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br,
                      score_scale_ + thread_id * Br, score_sum_ + thread_id * Br,
                      logsum_ + thread_id * Br, t_r_idx, t_c_idx, seq_size_q, seq_size_k,
                      causal_mask);

            // Rescale
            rescale_d(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q,
                      seq_size_k, causal_mask);

            // W @ V
            mma1_d(acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, head_size, dim_size, t_r_idx,
                   t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * Br * head_size * dim_size
                                    + this_thread_head * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * head_size * dim_size
                                    + Tc * Bc * head_size * dim_size + this_thread_head * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_d_n_fixed(Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size,
                           t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_d_n_fixed(Tc_left, acc_s_ + thread_id * Br * Bc,
                              acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                              scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                              score_sum_ + thread_id * Br, logsum_ + thread_id * Br, t_r_idx, Tc,
                              seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_d_n_fixed(Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc,
                              seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_d_n_fixed(Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, head_size,
                           dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_d(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
                                O + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * 1 * head_size * dim_size
                                    + this_thread_head * dim_size,
                                t_r_idx, head_size, dim_size);
        }
      }
    }
  }

  void fa2(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
           const dtype_t* __restrict__ V, dtype_t* __restrict__ O, const int32_t batch_size,
           const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
           const int32_t dim_size, bool causal_mask = true) {
    static_assert(Br == Bc);
    static_assert(Br % 4 == 0);
    assert(head_size % threads == 0);
    assert(dim_size % 8 == 0);

    // Prefill and Append mode
    if (seq_size_q != 1) {
      __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                           causal_mask);
    } else {
      __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                   causal_mask);
    }
  }

 private:
  // === Prefill and Append mode. Tile is always Br and Bc.
  // Br is small.
  // No need to use vector instr
  inline void init_temp(acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* acc_o,
                        const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < Br * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }

  // q_block  :Br x dim_size
  // k_block  :Bc x dim_size
  // acc_s    :Br x Bc
  inline void mma0(const dtype_t* __restrict__ q_block, const dtype_t* __restrict__ k_block,
                   acc_dtype_t* __restrict__ acc_s, const int32_t dim_size,
                   const int32_t stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                   const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * stride_size;
#pragma unroll
      for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

    // Delta for append mode.
    if (causal_mask && (global_r_end == global_c_end - delta_pos)) {
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < Bc; ++j) {
          if (j > i) { acc_s[i * Bc + j] = std::numeric_limits<float32_t>::lowest(); }
        }
      }
    }
  }

  inline void softmax(const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                      acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                      acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                      const int32_t t_c_idx, const int32_t seq_size_q, const int32_t seq_size_k,
                      bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

// 2. Reduce max(acc_s) to scoremax
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
// Vectorized max reduction
#pragma unroll
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
#pragma unroll
      for (int br = 0; br < Br; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br,
                  vexpq_fast_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    } else {  // High precession exp use libcall function from cmath.
#pragma unroll
      for (int br = 0; br < Br; ++br) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, vexpq_hp_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    }

// 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      score_sum[br] = sum;
    }

// 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
#pragma unroll
    for (int br = 0; br < Br; br += 4) {
      float32x4_t logsum_v = vld1q_f32(logsum + br);
      float32x4_t scale_v = vld1q_f32(score_scale + br);
      float32x4_t sum_v = vld1q_f32(score_sum + br);
      vst1q_f32(logsum + br, vmlaq_f32(sum_v, logsum_v, scale_v));
    }

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale(acc_dtype_t* __restrict__ acc_o, acc_dtype_t* __restrict__ score_scale,
                      const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                      const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

#pragma unroll
    for (int i = 0; i < Br; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is Br x Bc
  // v_block is Bc x dim_size
  inline void mma1(const dtype_t* __restrict__ w_block, const dtype_t* __restrict__ v_block,
                   acc_dtype_t* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                   const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                   const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

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

  inline void scale_and_cast_copy(const acc_dtype_t* __restrict__ acc_o,
                                  const acc_dtype_t* __restrict__ logsum,
                                  dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < Br; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Prefill and Append mode. Tile is Br_n_fixed and Bc_n_fixed(Original Br x Bc).
  // q_block  :Br_n_fixed x dim_size
  // k_block  :Bc_n_fixed x dim_size
  // acc_s    :Br_n_fixed x Bc_n_fixed
  inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                              const dtype_t* __restrict__ q_block,
                              const dtype_t* __restrict__ k_block, acc_dtype_t* __restrict__ acc_s,
                              const int32_t dim_size, const int32_t stride_size,
                              const int32_t t_r_idx, const int32_t t_c_idx,
                              const int32_t seq_size_q, const int32_t seq_size_k,
                              bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

    for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * stride_size;
      for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

    // Delta for append mode.
    if (causal_mask && (global_r_end == global_c_end - delta_pos)) {
      for (int i = 0; i < Br_n_fixed; ++i) {
        for (int j = 0; j < Bc_n_fixed; ++j) {
          if (j > i) { acc_s[i * Bc_n_fixed + j] = std::numeric_limits<float32_t>::lowest(); }
        }
      }
    }
  }

  inline void softmax_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                 const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                                 acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev,
                                 acc_dtype_t* score_scale, acc_dtype_t* score_sum,
                                 acc_dtype_t* logsum, const int32_t t_r_idx, const int32_t t_c_idx,
                                 const int32_t seq_size_q, const int32_t seq_size_k,
                                 bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

    // 2. Reduce max(acc_s) to scoremax
    for (int br = 0; br < Br_n_fixed; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
      // Vectorized max reduction
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      for (; bc < Bc_n_fixed; ++bc) { max_val = max_val > row[bc] ? max_val : row[bc]; }
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
      int br = 0;
      for (; br <= Br_n_fixed - 4; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br,
                  vexpq_fast_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
      for (; br < Br_n_fixed; ++br) {
        score_scale[br] = expf(scoremax_prev[br] * scale_ - scoremax[br] * scale_);
      }
    } else {  // High precession exp use libcall function from cmath.
      int br = 0;
      for (; br <= Br_n_fixed - 4; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, vexpq_hp_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
      for (; br < Br_n_fixed; ++br) {
        score_scale[br] = expf(scoremax_prev[br] * scale_ - scoremax[br] * scale_);
      }
    }

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    for (int br = 0; br < Br_n_fixed; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc_n_fixed;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      for (; bc < Bc_n_fixed; ++bc) {
        float exp = expf((row[bc] - sm) * scale_);
        row[bc] = exp;
        sum += exp;
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    int br = 0;
    for (; br <= Br_n_fixed - 4; br += 4) {
      float32x4_t logsum_v = vld1q_f32(logsum + br);
      float32x4_t scale_v = vld1q_f32(score_scale + br);
      float32x4_t sum_v = vld1q_f32(score_sum + br);
      vst1q_f32(logsum + br, vmlaq_f32(sum_v, logsum_v, scale_v));
    }
    for (; br < Br_n_fixed; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                 acc_dtype_t* __restrict__ acc_o,
                                 acc_dtype_t* __restrict__ score_scale, const int32_t dim_size,
                                 const int32_t t_r_idx, const int32_t t_c_idx,
                                 const int32_t seq_size_q, const int32_t seq_size_k,
                                 bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    for (int i = 0; i < Br_n_fixed; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is Br_n_fixed x Bc_n_fixed
  // v_block is Bc_n_fixed x dim_size
  inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                              const dtype_t* __restrict__ w_block,
                              const dtype_t* __restrict__ v_block, acc_dtype_t* __restrict__ acc_o,
                              const int32_t head_size, const int32_t dim_size,
                              const int32_t t_r_idx, const int32_t t_c_idx,
                              const int32_t seq_size_q, const int32_t seq_size_k,
                              bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

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

  inline void scale_and_cast_copy_pa_n_fixed(const int32_t Br_n_fixed,
                                             const acc_dtype_t* __restrict__ acc_o,
                                             const acc_dtype_t* __restrict__ logsum,
                                             dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                             const int32_t head_size, const int32_t dim_size) {
    for (int i = 0; i < Br_n_fixed; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Decode mode. Tile is always 1 and Bc.
  inline void init_temp_d(acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* acc_o,
                          const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < 1 * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }

  // q_block  :1 x dim_size
  // k_block  :Bc x dim_size
  // acc_s    :1 x Bc(Still use Br x Bc memory space, but others keeps empty)
  inline void mma0_d(const dtype_t* __restrict__ q_block, const dtype_t* __restrict__ k_block,
                     acc_dtype_t* __restrict__ acc_s, const int32_t dim_size,
                     const int32_t stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
    // There is no need to handle causal mask.
#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * stride_size;
#pragma unroll
      for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

  inline void softmax_d(const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                        acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                        acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                        const int32_t t_c_idx, const int32_t seq_size_q, const int32_t seq_size_k,
                        bool causal_mask) {
    // There is no need to handle causal mask in decoding stage.

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, 1 * sizeof(acc_dtype_t));

// 2. Reduce max(acc_s) to scoremax
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
// Vectorized max reduction
#pragma unroll
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    float32x4_t scale_vec = vdupq_n_f32(scale_);
#pragma unroll
    for (int br = 0; br < 1; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    logsum[0] = logsum[0] * score_scale[0] + score_sum[0];

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < 1 * Bc; i += 4) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float16x4_t v16 = vcvt_f16_f32(v0);
      vst1_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_d(acc_dtype_t* __restrict__ acc_o, acc_dtype_t* __restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
    // There is no need to handle causal mask.
#pragma unroll
    for (int i = 0; i < 1; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is 1 x Bc
  // v_block is Bc x dim_size
  inline void mma1_d(const dtype_t* __restrict__ w_block, const dtype_t* __restrict__ v_block,
                     acc_dtype_t* __restrict__ acc_o, const int32_t head_size,
                     const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#pragma unroll
    for (int b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
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

  inline void scale_and_cast_copy_d(const acc_dtype_t* __restrict__ acc_o,
                                    const acc_dtype_t* __restrict__ logsum,
                                    dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                    const int32_t head_size, const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < 1; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Decode mode. Tile is 1 x Bc_n_fixed(Original 1 x Bc)
  // q_block  :1 x dim_size
  // k_block  :Bc_n_fixed x dim_size
  // acc_s    :1 x Bc_n_fixed(Still use Br x Bc memory space, but others keeps empty)
  inline void mma0_d_n_fixed(const int32_t Bc_n_fixed, const dtype_t* __restrict__ q_block,
                             const dtype_t* __restrict__ k_block, acc_dtype_t* __restrict__ acc_s,
                             const int32_t dim_size, const int32_t stride_size,
                             const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                             const int32_t seq_size_k, bool causal_mask) {
// There is no need to handle causal mask.
#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * stride_size;
      for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

  inline void softmax_d_n_fixed(const int32_t Bc_n_fixed, const acc_dtype_t* __restrict__ acc_s,
                                dtype_t* acc_s_cast, acc_dtype_t* scoremax,
                                acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                                acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                                const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

    // 2. Reduce max(acc_s) to scoremax
    for (int br = 0; br < 1; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
      // Vectorized max reduction
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      for (; bc < Bc_n_fixed; ++bc) { max_val = max_val > row[bc] ? max_val : row[bc]; }
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
      score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);
    } else {  // High precession exp use libcall function from cmath.
      score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);
    }

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    for (int br = 0; br < 1; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc_n_fixed;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      for (; bc < Bc_n_fixed; ++bc) {
        float exp = expf((row[bc] - sm) * scale_);
        row[bc] = exp;
        sum += exp;
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    logsum[0] = logsum[0] * score_scale[0] + score_sum[0];

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t* __restrict__ acc_o,
                                acc_dtype_t* __restrict__ score_scale, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
    for (int i = 0; i < 1; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  inline void mma1_d_n_fixed(const int32_t Bc_n_fixed, const dtype_t* __restrict__ w_block,
                             const dtype_t* __restrict__ v_block, acc_dtype_t* __restrict__ acc_o,
                             const int32_t head_size, const int32_t dim_size, const int32_t t_r_idx,
                             const int32_t t_c_idx, const int32_t seq_size_q,
                             const int32_t seq_size_k, bool causal_mask) {
    for (int b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
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

  float scale_;
  dtype_t* acc_s_cast_;         // [threads][Br * Bc]
  acc_dtype_t* acc_o_;          // [threads][Br * dim_size]
  acc_dtype_t* acc_s_;          // [threads][Br * Bc]
  acc_dtype_t* logsum_;         // [threads][Br]
  acc_dtype_t* scoremax_;       // [threads][Br]
  acc_dtype_t* scoremax_prev_;  // [threads][Br]
  acc_dtype_t* score_scale_;    // [threads][Br]
  acc_dtype_t* score_sum_;      // [threads][Br]
};

template<int32_t Br, int32_t Bc, int32_t Q_Head, int32_t KV_Head, int32_t threads = 4,
         bool HP = false>
struct NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL {
  using dtype_t = float16_t;
  using acc_dtype_t = float32_t;

  void init_workspace(dtype_t* acc_s_cast, acc_dtype_t* acc_o, acc_dtype_t* acc_s,
                      acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev,
                      acc_dtype_t* score_scale, acc_dtype_t* score_sum) {
    acc_s_cast_ = acc_s_cast;        // [threads][Br * Bc]
    acc_o_ = acc_o;                  // [threads][Br * dim_size]
    acc_s_ = acc_s;                  // [threads][Br * Bc]
    logsum_ = logsum;                // [threads][Br]
    scoremax_ = scoremax;            // [threads][Br]
    scoremax_prev_ = scoremax_prev;  // [threads][Br]
    score_scale_ = score_scale;      // [threads][Br]
    score_sum_ = score_sum;          // [threads][Br]
  }

  // For prefill case.
  inline void __fa2_prefill_append(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
                                   const dtype_t* __restrict__ V, dtype_t* __restrict__ O,
                                   const int32_t batch_size, const int32_t head_size,
                                   const int32_t seq_size_q, const int32_t seq_size_k,
                                   const int32_t dim_size, bool causal_mask = true) {
    // Constant
    const int32_t Tr = seq_size_q / Br;
    const int32_t Tr_left = seq_size_q % Br;
    const int32_t Tc = seq_size_k / Bc;
    const int32_t Tc_left = seq_size_k % Bc;

    scale_ = sqrt(1.0 / dim_size);

    // Loops
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME: Make dynamic threads dispatching available in head_size level.
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
      for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
        const int32_t thread_id = omp_get_thread_num();
        const int32_t this_thread_head = h_idx;
        const int32_t this_thread_head_q = this_thread_head;
        const int32_t this_thread_head_kv = kv_head_index(this_thread_head_q);

        // Loop S_Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                    acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + t_r_idx * Br * Q_Head * dim_size
                                    + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0(tile_q, tile_k, tile_acc_s, dim_size, Q_Head * dim_size, KV_Head * dim_size,
                 t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax(acc_s_ + thread_id * Br * Bc, acc_s_cast_ + thread_id * Br * Bc,
                    scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br,
                    score_scale_ + thread_id * Br, score_sum_ + thread_id * Br,
                    logsum_ + thread_id * Br, t_r_idx, t_c_idx, seq_size_q, seq_size_k,
                    causal_mask);

            // Rescale
            rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q,
                    seq_size_k, causal_mask);

            // W @ V
            mma1(acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, KV_Head, dim_size, t_r_idx,
                 t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          // Process the last block of KV
          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + t_r_idx * Br * Q_Head * dim_size
                                    + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_pa_n_fixed(Br, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, Q_Head * dim_size,
                            KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_pa_n_fixed(Br, Tc_left, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, t_r_idx, Tc,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Br, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx,
                               Tc, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Br, Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, KV_Head,
                            dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
                              O + b_idx * seq_size_q * head_size * dim_size
                                  + t_r_idx * Br * head_size * dim_size
                                  + this_thread_head * dim_size,
                              t_r_idx, head_size, dim_size);
        }

        // Process left Q
        if (Tr_left) {
          // Init all temps
          init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                    acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + Tr * Br * Q_Head * dim_size + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0_pa_n_fixed(Tr_left, Bc, tile_q, tile_k, tile_acc_s, dim_size, Q_Head * dim_size,
                            KV_Head * dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_pa_n_fixed(Tr_left, Bc, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, Tr, t_c_idx,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Tr_left, Bc, acc_o, score_scale_ + thread_id * Br, dim_size, Tr,
                               t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Tr_left, Bc, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, KV_Head,
                            dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          // Process the last block of KV
          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + Tr * Br * Q_Head * dim_size + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_pa_n_fixed(Tr_left, Tc_left, tile_q, tile_k, tile_acc_s, dim_size,
                            Q_Head * dim_size, KV_Head * dim_size, Tr, Tc, seq_size_q, seq_size_k,
                            causal_mask);

            // Softmax
            softmax_pa_n_fixed(Tr_left, Tc_left, acc_s_ + thread_id * Br * Bc,
                               acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                               scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                               score_sum_ + thread_id * Br, logsum_ + thread_id * Br, Tr, Tc,
                               seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_pa_n_fixed(Tr_left, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, Tr,
                               Tc, seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_pa_n_fixed(Tr_left, Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o,
                            KV_Head, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_pa_n_fixed(
              Tr_left, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
              O + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size
                  + this_thread_head * dim_size,
              Tr, head_size, dim_size);
        }
      }
    }
  }

  // For decode case.
  inline void __fa2_decode(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
                           const dtype_t* __restrict__ V, dtype_t* __restrict__ O,
                           const int32_t batch_size, const int32_t head_size,
                           const int32_t seq_size_q, const int32_t seq_size_k,
                           const int32_t dim_size, bool causal_mask = true) {
    // Constant
    const int32_t Tr = 1;
    const int32_t Tc = seq_size_k / Bc;
    const int32_t Tc_left = seq_size_k % Bc;
    // scale_ = sqrt(1.0 / dim_size) * 1.44269504;
    scale_ = sqrt(1.0 / dim_size);

    // Loops
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME: Make dynamic threads dispatching available in head_size level.
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
      for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
        const int32_t thread_id = omp_get_thread_num();
        const int32_t this_thread_head = h_idx;
        const int32_t this_thread_head_q = this_thread_head;
        const int32_t this_thread_head_kv = kv_head_index(this_thread_head_q);

        // Loop Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp_d(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                      acc_o_ + thread_id * Br * dim_size, dim_size);

          // Loop KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + t_r_idx * 1 * Q_Head * dim_size
                                    + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + t_c_idx * Bc * KV_Head * dim_size
                                    + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * 1 * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;
            // Q @ K^T
            mma0_d(tile_q, tile_k, tile_acc_s, dim_size, Q_Head * dim_size, KV_Head * dim_size,
                   t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_d(acc_s_ + thread_id * Br * Bc, acc_s_cast_ + thread_id * Br * Bc,
                      scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br,
                      score_scale_ + thread_id * Br, score_sum_ + thread_id * Br,
                      logsum_ + thread_id * Br, t_r_idx, t_c_idx, seq_size_q, seq_size_k,
                      causal_mask);

            // Rescale
            rescale_d(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q,
                      seq_size_k, causal_mask);

            // W @ V
            mma1_d(acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, KV_Head, dim_size, t_r_idx,
                   t_c_idx, seq_size_q, seq_size_k, causal_mask);
          }

          if (Tc_left) {
            const dtype_t* tile_q = Q + b_idx * seq_size_q * Q_Head * dim_size
                                    + t_r_idx * Br * Q_Head * dim_size
                                    + this_thread_head_q * dim_size;
            const dtype_t* tile_k = K + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            const dtype_t* tile_v = V + b_idx * seq_size_k * KV_Head * dim_size
                                    + Tc * Bc * KV_Head * dim_size + this_thread_head_kv * dim_size;
            acc_dtype_t* tile_acc_s = acc_s_ + thread_id * Br * Bc;
            acc_dtype_t* acc_o = acc_o_ + thread_id * Br * dim_size;

            // Q @ K^T
            mma0_d_n_fixed(Tc_left, tile_q, tile_k, tile_acc_s, dim_size, Q_Head * dim_size,
                           KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);

            // Softmax
            softmax_d_n_fixed(Tc_left, acc_s_ + thread_id * Br * Bc,
                              acc_s_cast_ + thread_id * Br * Bc, scoremax_ + thread_id * Br,
                              scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br,
                              score_sum_ + thread_id * Br, logsum_ + thread_id * Br, t_r_idx, Tc,
                              seq_size_q, seq_size_k, causal_mask);

            // Rescale
            rescale_d_n_fixed(Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc,
                              seq_size_q, seq_size_k, causal_mask);

            // W @ V
            mma1_d_n_fixed(Tc_left, acc_s_cast_ + thread_id * Br * Bc, tile_v, acc_o, KV_Head,
                           dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_d(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br,
                                O + b_idx * seq_size_q * head_size * dim_size
                                    + t_r_idx * 1 * head_size * dim_size
                                    + this_thread_head * dim_size,
                                t_r_idx, head_size, dim_size);
        }
      }
    }
  }

  void fa2(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
           const dtype_t* __restrict__ V, dtype_t* __restrict__ O, const int32_t batch_size,
           const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
           const int32_t dim_size, bool causal_mask = true) {
    static_assert(Br == Bc);
    static_assert(Br % 4 == 0);
    static_assert(Q_Head % KV_Head == 0);
    assert(head_size % threads == 0);
    assert(dim_size % 8 == 0);

    // Prefill and Append mode
    if (seq_size_q != 1) {
      __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                           causal_mask);
    } else {
      __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                   causal_mask);
    }
  }

 private:
  // === Prefill and Append mode. Tile is always Br and Bc.
  // Br is small.
  // No need to use vector instr
  inline void init_temp(acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* acc_o,
                        const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < Br * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }

  inline int32_t kv_head_index(int32_t q_head_index) { return q_head_index / (Q_Head / KV_Head); }

  // q_block  :Br x dim_size
  // k_block  :Bc x dim_size
  // acc_s    :Br x Bc
  inline void mma0(const dtype_t* __restrict__ q_block, const dtype_t* __restrict__ k_block,
                   acc_dtype_t* __restrict__ acc_s, const int32_t dim_size,
                   const int32_t q_stride_size, const int32_t kv_stride_size, const int32_t t_r_idx,
                   const int32_t t_c_idx, const int32_t seq_size_q, const int32_t seq_size_k,
                   bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * q_stride_size;
#pragma unroll
      for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * kv_stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

    // Delta for append mode.
    if (causal_mask && (global_r_end == global_c_end - delta_pos)) {
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < Bc; ++j) {
          if (j > i) { acc_s[i * Bc + j] = std::numeric_limits<float32_t>::lowest(); }
        }
      }
    }
  }

  inline void softmax(const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                      acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                      acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                      const int32_t t_c_idx, const int32_t seq_size_q, const int32_t seq_size_k,
                      bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

// 2. Reduce max(acc_s) to scoremax
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
// Vectorized max reduction
#pragma unroll
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
#pragma unroll
      for (int br = 0; br < Br; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br,
                  vexpq_fast_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    } else {  // High precession exp use libcall function from cmath.
#pragma unroll
      for (int br = 0; br < Br; ++br) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, vexpq_hp_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    }

// 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      score_sum[br] = sum;
    }

// 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
#pragma unroll
    for (int br = 0; br < Br; br += 4) {
      float32x4_t logsum_v = vld1q_f32(logsum + br);
      float32x4_t scale_v = vld1q_f32(score_scale + br);
      float32x4_t sum_v = vld1q_f32(score_sum + br);
      vst1q_f32(logsum + br, vmlaq_f32(sum_v, logsum_v, scale_v));
    }

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale(acc_dtype_t* __restrict__ acc_o, acc_dtype_t* __restrict__ score_scale,
                      const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                      const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

#pragma unroll
    for (int i = 0; i < Br; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is Br x Bc
  // v_block is Bc x dim_size
  inline void mma1(const dtype_t* __restrict__ w_block, const dtype_t* __restrict__ v_block,
                   acc_dtype_t* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                   const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                   const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

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

  inline void scale_and_cast_copy(const acc_dtype_t* __restrict__ acc_o,
                                  const acc_dtype_t* __restrict__ logsum,
                                  dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < Br; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Prefill and Append mode. Tile is Br_n_fixed and Bc_n_fixed(Original Br x Bc).
  // q_block  :Br_n_fixed x dim_size
  // k_block  :Bc_n_fixed x dim_size
  // acc_s    :Br_n_fixed x Bc_n_fixed
  inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                              const dtype_t* __restrict__ q_block,
                              const dtype_t* __restrict__ k_block, acc_dtype_t* __restrict__ acc_s,
                              const int32_t dim_size, const int32_t q_stride_size,
                              const int32_t kv_stride_size, const int32_t t_r_idx,
                              const int32_t t_c_idx, const int32_t seq_size_q,
                              const int32_t seq_size_k, bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

    for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * q_stride_size;
      for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * kv_stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

    // Delta for append mode.
    if (causal_mask && (global_r_end == global_c_end - delta_pos)) {
      for (int i = 0; i < Br_n_fixed; ++i) {
        for (int j = 0; j < Bc_n_fixed; ++j) {
          if (j > i) { acc_s[i * Bc_n_fixed + j] = std::numeric_limits<float32_t>::lowest(); }
        }
      }
    }
  }

  inline void softmax_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                 const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                                 acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev,
                                 acc_dtype_t* score_scale, acc_dtype_t* score_sum,
                                 acc_dtype_t* logsum, const int32_t t_r_idx, const int32_t t_c_idx,
                                 const int32_t seq_size_q, const int32_t seq_size_k,
                                 bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

    // 2. Reduce max(acc_s) to scoremax
    for (int br = 0; br < Br_n_fixed; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
      // Vectorized max reduction
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      for (; bc < Bc_n_fixed; ++bc) { max_val = max_val > row[bc] ? max_val : row[bc]; }
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
      int br = 0;
      for (; br <= Br_n_fixed - 4; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br,
                  vexpq_fast_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
      for (; br < Br_n_fixed; ++br) {
        score_scale[br] = expf(scoremax_prev[br] * scale_ - scoremax[br] * scale_);
      }
    } else {  // High precession exp use libcall function from cmath.
      int br = 0;
      for (; br <= Br_n_fixed - 4; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, vexpq_hp_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
      for (; br < Br_n_fixed; ++br) {
        score_scale[br] = expf(scoremax_prev[br] * scale_ - scoremax[br] * scale_);
      }
    }

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    for (int br = 0; br < Br_n_fixed; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc_n_fixed;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      for (; bc < Bc_n_fixed; ++bc) {
        float exp = expf((row[bc] - sm) * scale_);
        row[bc] = exp;
        sum += exp;
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    int br = 0;
    for (; br <= Br_n_fixed - 4; br += 4) {
      float32x4_t logsum_v = vld1q_f32(logsum + br);
      float32x4_t scale_v = vld1q_f32(score_scale + br);
      float32x4_t sum_v = vld1q_f32(score_sum + br);
      vst1q_f32(logsum + br, vmlaq_f32(sum_v, logsum_v, scale_v));
    }
    for (; br < Br_n_fixed; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                 acc_dtype_t* __restrict__ acc_o,
                                 acc_dtype_t* __restrict__ score_scale, const int32_t dim_size,
                                 const int32_t t_r_idx, const int32_t t_c_idx,
                                 const int32_t seq_size_q, const int32_t seq_size_k,
                                 bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

    for (int i = 0; i < Br_n_fixed; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is Br_n_fixed x Bc_n_fixed
  // v_block is Bc_n_fixed x dim_size
  inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                              const dtype_t* __restrict__ w_block,
                              const dtype_t* __restrict__ v_block, acc_dtype_t* __restrict__ acc_o,
                              const int32_t head_size, const int32_t dim_size,
                              const int32_t t_r_idx, const int32_t t_c_idx,
                              const int32_t seq_size_q, const int32_t seq_size_k,
                              bool causal_mask) {
    const int32_t global_r_start = t_r_idx * Br;
    const int32_t global_r_end =
        global_r_start + Br < seq_size_q ? global_r_start + Br : seq_size_q;
    const int32_t global_c_start = t_c_idx * Bc;
    const int32_t global_c_end =
        global_c_start + Bc < seq_size_k ? global_c_start + Bc : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) return;

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

  inline void scale_and_cast_copy_pa_n_fixed(const int32_t Br_n_fixed,
                                             const acc_dtype_t* __restrict__ acc_o,
                                             const acc_dtype_t* __restrict__ logsum,
                                             dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                             const int32_t head_size, const int32_t dim_size) {
    for (int i = 0; i < Br_n_fixed; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Decode mode. Tile is always 1 and Bc.
  inline void init_temp_d(acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* acc_o,
                          const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);
#pragma unroll
    for (int i = 0; i < Br; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < 1 * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }

  // q_block  :1 x dim_size
  // k_block  :Bc x dim_size
  // acc_s    :1 x Bc(Still use Br x Bc memory space, but others keeps empty)
  inline void mma0_d(const dtype_t* __restrict__ q_block, const dtype_t* __restrict__ k_block,
                     acc_dtype_t* __restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
// There is no need to handle causal mask.
#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * q_stride_size;
#pragma unroll
      for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * kv_stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

  inline void softmax_d(const acc_dtype_t* __restrict__ acc_s, dtype_t* acc_s_cast,
                        acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                        acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                        const int32_t t_c_idx, const int32_t seq_size_q, const int32_t seq_size_k,
                        bool causal_mask) {
    // There is no need to handle causal mask in decoding stage.

    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, 1 * sizeof(acc_dtype_t));

// 2. Reduce max(acc_s) to scoremax
#pragma unroll
    for (int br = 0; br < Br; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
// Vectorized max reduction
#pragma unroll
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    float32x4_t scale_vec = vdupq_n_f32(scale_);
#pragma unroll
    for (int br = 0; br < 1; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      for (int bc = 0; bc < Bc; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    logsum[0] = logsum[0] * score_scale[0] + score_sum[0];

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < 1 * Bc; i += 4) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float16x4_t v16 = vcvt_f16_f32(v0);
      vst1_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_d(acc_dtype_t* __restrict__ acc_o, acc_dtype_t* __restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
// There is no need to handle causal mask.
#pragma unroll
    for (int i = 0; i < 1; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  // w_block is 1 x Bc
  // v_block is Bc x dim_size
  inline void mma1_d(const dtype_t* __restrict__ w_block, const dtype_t* __restrict__ v_block,
                     acc_dtype_t* __restrict__ acc_o, const int32_t head_size,
                     const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#pragma unroll
    for (int b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
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

  inline void scale_and_cast_copy_d(const acc_dtype_t* __restrict__ acc_o,
                                    const acc_dtype_t* __restrict__ logsum,
                                    dtype_t* __restrict__ o_block, const int32_t t_r_idx,
                                    const int32_t head_size, const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < 1; ++i) {
      float16_t* o_block_line = o_block + i * head_size * dim_size;
      float reciprocal_logsum = 1.0f / logsum[i];

      int j = 0;
      for (; j <= dim_size - 8; j += 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o + i * dim_size + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o + i * dim_size + j + 4);
        float32x4_t vec_reciprocal_logsum = vdupq_n_f32(reciprocal_logsum);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal_logsum);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal_logsum);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
      }

      for (; j < dim_size; ++j) {
        o_block_line[j] = (float16_t)(acc_o[i * dim_size + j] * reciprocal_logsum);
      }
    }
  }

  // === Decode mode. Tile is 1 x Bc_n_fixed(Original 1 x Bc)
  // q_block  :1 x dim_size
  // k_block  :Bc_n_fixed x dim_size
  // acc_s    :1 x Bc_n_fixed(Still use Br x Bc memory space, but others keeps empty)
  inline void mma0_d_n_fixed(const int32_t Bc_n_fixed, const dtype_t* __restrict__ q_block,
                             const dtype_t* __restrict__ k_block, acc_dtype_t* __restrict__ acc_s,
                             const int32_t dim_size, const int32_t q_stride_size,
                             const int32_t kv_stride_size, const int32_t t_r_idx,
                             const int32_t t_c_idx, const int32_t seq_size_q,
                             const int32_t seq_size_k, bool causal_mask) {
// There is no need to handle causal mask.
#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
      const dtype_t* q_block_line = q_block + b_r_idx * q_stride_size;
      for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
        const dtype_t* k_block_line = k_block + b_c_idx * kv_stride_size;

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
        acc_dtype_t total =
            vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

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

  inline void softmax_d_n_fixed(const int32_t Bc_n_fixed, const acc_dtype_t* __restrict__ acc_s,
                                dtype_t* acc_s_cast, acc_dtype_t* scoremax,
                                acc_dtype_t* scoremax_prev, acc_dtype_t* score_scale,
                                acc_dtype_t* score_sum, acc_dtype_t* logsum, const int32_t t_r_idx,
                                const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

    // 2. Reduce max(acc_s) to scoremax
    for (int br = 0; br < 1; ++br) {
      float32x4_t max_vec = vdupq_n_f32(NEG_INF);
      const acc_dtype_t* row = acc_s + br * Bc;
      // Vectorized max reduction
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      for (; bc < Bc_n_fixed; ++bc) { max_val = max_val > row[bc] ? max_val : row[bc]; }
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale_);
    if constexpr (!HP) {  // Use approximate method to calculate exp.
      score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);
    } else {  // High precession exp use libcall function from cmath.
      score_scale[0] = expf(scoremax_prev[0] * scale_ - scoremax[0] * scale_);
    }

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    for (int br = 0; br < 1; ++br) {
      const float sm = scoremax[br];
      acc_dtype_t* row = const_cast<acc_dtype_t*>(acc_s) + br * Bc_n_fixed;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      int bc = 0;
      for (; bc <= Bc_n_fixed - 4; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP) {
          exp_vec = vexpq_fast_f32(val_vec);
        } else {
          exp_vec = vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      for (; bc < Bc_n_fixed; ++bc) {
        float exp = expf((row[bc] - sm) * scale_);
        row[bc] = exp;
        sum += exp;
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    logsum[0] = logsum[0] * score_scale[0] + score_sum[0];

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < Br * Bc; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  inline void rescale_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t* __restrict__ acc_o,
                                acc_dtype_t* __restrict__ score_scale, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
    for (int i = 0; i < 1; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      for (int j = 0; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  inline void mma1_d_n_fixed(const int32_t Bc_n_fixed, const dtype_t* __restrict__ w_block,
                             const dtype_t* __restrict__ v_block, acc_dtype_t* __restrict__ acc_o,
                             const int32_t head_size, const int32_t dim_size, const int32_t t_r_idx,
                             const int32_t t_c_idx, const int32_t seq_size_q,
                             const int32_t seq_size_k, bool causal_mask) {
    for (int b_r_idx = 0; b_r_idx < 1; ++b_r_idx) {
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

  float scale_;
  dtype_t* acc_s_cast_;         // [threads][Br * Bc]
  acc_dtype_t* acc_o_;          // [threads][Br * dim_size]
  acc_dtype_t* acc_s_;          // [threads][Br * Bc]
  acc_dtype_t* logsum_;         // [threads][Br]
  acc_dtype_t* scoremax_;       // [threads][Br]
  acc_dtype_t* scoremax_prev_;  // [threads][Br]
  acc_dtype_t* score_scale_;    // [threads][Br]
  acc_dtype_t* score_sum_;      // [threads][Br]
};

template<typename FlashAttn2Impl>
struct FlashAttn2 {
 public:
  using dtype_t = FlashAttn2Impl::dtype_t;
  using acc_dtype_t = FlashAttn2Impl::acc_dtype_t;

  inline FlashAttn2Impl& __impl() { return fa2_impl_; }

  inline void init_workspace(dtype_t* acc_s_cast, acc_dtype_t* acc_o, acc_dtype_t* acc_s,
                             acc_dtype_t* logsum, acc_dtype_t* scoremax, acc_dtype_t* scoremax_prev,
                             acc_dtype_t* score_scale, acc_dtype_t* score_sum) {
    __impl().init_workspace(acc_s_cast, acc_o, acc_s, logsum, scoremax, scoremax_prev, score_scale,
                            score_sum);
  }

  inline void operator()(const dtype_t* __restrict__ Q, const dtype_t* __restrict__ K,
                         const dtype_t* __restrict__ V, dtype_t* __restrict__ O,
                         const int32_t batch_size, const int32_t head_size,
                         const int32_t seq_size_q, const int32_t seq_size_k, const int32_t dim_size,
                         bool causal_mask = true) {
    __impl().fa2(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
  }

 private:
  FlashAttn2Impl fa2_impl_;
};

}  // namespace mobi_attn

#endif
