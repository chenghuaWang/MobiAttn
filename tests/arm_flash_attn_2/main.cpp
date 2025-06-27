#include <cassert>
#include <random>
#include <arm_neon.h>
#include <gtest/gtest.h>
#include "mobi_attn/flash_attn_2/driver.hpp"
#include "mobi_attn/flash_attn_2/arm_qkv_fp16_o_fp16_fa2.hpp"

using namespace mobi_attn;

void arm_align_alloc(void** ptr, size_t required_bytes, size_t align) {
  if (align == 0 || (align & (align - 1))) {
    *ptr = nullptr;
    return;
  }
  void* p1;
  void** p2;
  size_t offset = align - 1 + sizeof(void*);
  if ((p1 = (void*)malloc(required_bytes + offset)) == nullptr) {
    *ptr = nullptr;
    return;
  }
  p2 = (void**)(((size_t)(p1) + offset) & ~(align - 1));
  p2[-1] = p1;
  *ptr = p2;
  assert(reinterpret_cast<size_t>(*ptr) % align == 0);
}

void arm_align_free(void* ptr) { free(((void**)ptr)[-1]); }

void causal_attention_mha(float* output, const float* Q, const float* K, const float* V, int B,
                          int S_Q, int S_K, int H, int D) {
  const float L_INF = std::numeric_limits<float>::lowest();
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float* attn_scores = new float[S_Q * S_K];

      for (int i = 0; i < S_Q; ++i) {
        for (int j = 0; j < S_K; ++j) {
          float score = 0.0f;
          for (int k = 0; k < D; ++k) {
            int q_idx = ((b * S_Q + i) * H + h) * D + k;
            int k_idx = ((b * S_K + j) * H + h) * D + k;
            score += Q[q_idx] * K[k_idx];
          }
          score *= scale;
          if (S_Q != 1 && S_Q == S_K && j > i) {
            score = L_INF;
          } else if (S_Q != 1 && S_Q < S_K && j - (S_K - S_Q) > i) {
            score = L_INF;
          }
          attn_scores[i * S_K + j] = score;
        }
      }

      // Softmax
      for (int i = 0; i < S_Q; ++i) {
        float max_val = L_INF;
        for (int j = 0; j < S_K; ++j) {
          if (attn_scores[i * S_K + j] > max_val) max_val = attn_scores[i * S_K + j];
        }

        float exp_sum = 0.0f;
        for (int j = 0; j < S_K; ++j) {
          float exp_val = expf(attn_scores[i * S_K + j] - max_val);
          attn_scores[i * S_K + j] = exp_val;
          exp_sum += exp_val;
        }

        for (int j = 0; j < S_K; ++j) attn_scores[i * S_K + j] /= exp_sum;
      }

      for (int i = 0; i < S_Q; ++i) {
        for (int d = 0; d < D; ++d) {
          float val = 0.0f;
          for (int j = 0; j < S_K; ++j) {
            int v_idx = ((b * S_K + j) * H + h) * D + d;
            val += attn_scores[i * S_K + j] * V[v_idx];
          }
          int out_idx = ((b * S_Q + i) * H + h) * D + d;
          output[out_idx] = val;
        }
      }
      delete[] attn_scores;
    }
  }
}

void causal_attention_gqa(float* output, const float* Q, const float* K, const float* V, int B,
                          int S_Q, int S_K, int H_q, int H_kv, int D) {
  const float L_INF = std::numeric_limits<float>::lowest();
  const float scale = 1.0f / sqrtf(static_cast<float>(D));
  const int G = H_q / H_kv;

  for (int b = 0; b < B; ++b) {
    for (int h_q = 0; h_q < H_q; ++h_q) {
      const int h_kv = h_q / G;
      float* attn_scores = new float[S_Q * S_K];

      for (int i = 0; i < S_Q; ++i) {
        for (int j = 0; j < S_K; ++j) {
          float score = 0.0f;
          for (int k = 0; k < D; ++k) {
            const int q_idx = ((b * S_Q + i) * H_q + h_q) * D + k;
            const int k_idx = ((b * S_K + j) * H_kv + h_kv) * D + k;
            score += Q[q_idx] * K[k_idx];
          }
          score *= scale;

          if (S_Q != 1) {
            if (S_Q == S_K && j > i) {
              score = L_INF;
            } else if (S_Q < S_K) {
              const int offset = S_K - S_Q;
              if (j - offset > i) { score = L_INF; }
            }
          }
          attn_scores[i * S_K + j] = score;
        }
      }

      for (int i = 0; i < S_Q; ++i) {
        float max_val = L_INF;
        for (int j = 0; j < S_K; ++j) {
          if (attn_scores[i * S_K + j] > max_val) { max_val = attn_scores[i * S_K + j]; }
        }

        float exp_sum = 0.0f;
        for (int j = 0; j < S_K; ++j) {
          const float exp_val = expf(attn_scores[i * S_K + j] - max_val);
          attn_scores[i * S_K + j] = exp_val;
          exp_sum += exp_val;
        }

        for (int j = 0; j < S_K; ++j) { attn_scores[i * S_K + j] /= exp_sum; }
      }

      for (int i = 0; i < S_Q; ++i) {
        for (int d = 0; d < D; ++d) {
          float val = 0.0f;
          for (int j = 0; j < S_K; ++j) {
            const int v_idx = ((b * S_K + j) * H_kv + h_kv) * D + d;
            val += attn_scores[i * S_K + j] * V[v_idx];
          }
          const int out_idx = ((b * S_Q + i) * H_q + h_q) * D + d;
          output[out_idx] = val;
        }
      }
      delete[] attn_scores;
    }
  }
}

class ArmFlashAttn2_MHA_BSHD_FP16_Test : public testing::Test {
 protected:
  ArmFlashAttn2_MHA_BSHD_FP16_Test() = default;

  ~ArmFlashAttn2_MHA_BSHD_FP16_Test() override = default;

  using FlashAttnOp =
      FlashAttn2<NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<4, 4, 2, false>>;

  void SetShape(int B, int S_Q, int S_K, int H, int D) {
    B_ = B;
    S_Q_ = S_Q;
    S_K_ = S_K;
    H_ = H;
    D_ = D;
  }

  void AllocAndInitFACtx() {
    constexpr int32_t Br = 4;
    constexpr int32_t Bc = 4;
    constexpr int32_t threads = 2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    arm_align_alloc((void**)&Q_fp32_, B_ * S_Q_ * H_ * D_ * sizeof(float), 16);
    arm_align_alloc((void**)&K_fp32_, B_ * S_K_ * H_ * D_ * sizeof(float), 16);
    arm_align_alloc((void**)&V_fp32_, B_ * S_K_ * H_ * D_ * sizeof(float), 16);

    for (int i = 0; i < B_ * S_Q_ * H_ * D_; ++i) { Q_fp32_[i] = dist(gen); }
    for (int i = 0; i < B_ * S_K_ * H_ * D_; ++i) {
      K_fp32_[i] = dist(gen);
      V_fp32_[i] = dist(gen);
    }

    arm_align_alloc((void**)&Q_h_, B_ * S_Q_ * H_ * D_ * sizeof(float16_t), 16);
    arm_align_alloc((void**)&K_h_, B_ * S_K_ * H_ * D_ * sizeof(float16_t), 16);
    arm_align_alloc((void**)&V_h_, B_ * S_K_ * H_ * D_ * sizeof(float16_t), 16);

    for (int i = 0; i < B_ * S_Q_ * H_ * D_; ++i) { Q_h_[i] = static_cast<float16_t>(Q_fp32_[i]); }
    for (int i = 0; i < B_ * S_K_ * H_ * D_; ++i) {
      K_h_[i] = static_cast<float16_t>(K_fp32_[i]);
      V_h_[i] = static_cast<float16_t>(V_fp32_[i]);
    }

    arm_align_alloc((void**)&output_causal_, B_ * S_Q_ * H_ * D_ * sizeof(float), 16);
    memset(output_causal_, 0, B_ * S_Q_ * H_ * D_ * sizeof(float));

    arm_align_alloc((void**)&O_h_, B_ * S_Q_ * H_ * D_ * sizeof(float16_t), 16);

    arm_align_alloc(&acc_s_cast_, threads * Br * Bc * sizeof(FlashAttnOp::dtype_t), 16);
    arm_align_alloc(&acc_o_, threads * Br * D_ * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&acc_s_, threads * Br * Bc * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&logsum_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&scoremax_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&scoremax_prev_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&score_scale_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&score_sum_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);

    arm_align_alloc((void**)&output_flash_, B_ * S_Q_ * H_ * D_ * sizeof(float16_t), 16);

    fa_op_ = FlashAttnOp();
    fa_op_.init_workspace(
        (FlashAttnOp::dtype_t*)acc_s_cast_, (FlashAttnOp::acc_dtype_t*)acc_o_,
        (FlashAttnOp::acc_dtype_t*)acc_s_, (FlashAttnOp::acc_dtype_t*)logsum_,
        (FlashAttnOp::acc_dtype_t*)scoremax_, (FlashAttnOp::acc_dtype_t*)scoremax_prev_,
        (FlashAttnOp::acc_dtype_t*)score_scale_, (FlashAttnOp::acc_dtype_t*)score_sum_);
  }

  void CalculateRef() {
    causal_attention_mha(output_causal_, Q_fp32_, K_fp32_, V_fp32_, B_, S_Q_, S_K_, H_, D_);
  }

  void Calculate() {
    fa_op_(Q_h_, K_h_, V_h_, O_h_, B_, H_, H_, S_Q_, S_K_, D_);
    for (int i = 0; i < B_ * S_Q_ * H_ * D_; ++i) { output_flash_[i] = (float)(O_h_[i]); }
  }

  bool Compare() {
    const float atol = 1e-2f;
    const float rtol = 1e-2f;
    bool all_close = true;
    float max_diff = 0.0f;

    for (int i = 0; i < B_ * S_Q_ * H_ * D_; ++i) {
      float diff = std::abs(output_causal_[i] - output_flash_[i]);
      float a = std::abs(output_causal_[i]);
      float b = std::abs(output_flash_[i]);

      if (diff > atol + rtol * std::max(a, b)) {
        all_close = false;
        if (diff > max_diff) max_diff = diff;
      }
    }

    return all_close;
  }

  void TearDown() override {
    arm_align_free(Q_fp32_);
    arm_align_free(K_fp32_);
    arm_align_free(V_fp32_);
    arm_align_free(Q_h_);
    arm_align_free(K_h_);
    arm_align_free(V_h_);
    arm_align_free(output_causal_);
    arm_align_free(O_h_);
    arm_align_free(output_flash_);
    arm_align_free(acc_s_cast_);
    arm_align_free(acc_o_);
    arm_align_free(acc_s_);
    arm_align_free(logsum_);
    arm_align_free(scoremax_);
    arm_align_free(scoremax_prev_);
    arm_align_free(score_scale_);
    arm_align_free(score_sum_);
  }

 private:
  FlashAttnOp fa_op_;

  int B_;
  int S_Q_;
  int S_K_;
  int H_;
  int D_;

  float *Q_fp32_, *K_fp32_, *V_fp32_;
  float16_t *Q_h_, *K_h_, *V_h_, *O_h_;
  float* output_flash_;
  float* output_causal_;

  void* acc_s_cast_;
  void* acc_o_;
  void* acc_s_;
  void* logsum_;
  void* scoremax_;
  void* scoremax_prev_;
  void* score_scale_;
  void* score_sum_;
};

class ArmFlashAttn2_GQA_BSHD_FP16_Test : public testing::Test {
 protected:
  ArmFlashAttn2_GQA_BSHD_FP16_Test() = default;

  ~ArmFlashAttn2_GQA_BSHD_FP16_Test() override = default;

  using FlashAttnOp =
      FlashAttn2<NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<4, 4, 2, false, true>>;

  void SetShape(int B, int S_Q, int S_K, int D) {
    B_ = B;
    S_Q_ = S_Q;
    S_K_ = S_K;
    D_ = D;
  }

  void AllocAndInitFACtx() {
    constexpr int32_t Br = 4;
    constexpr int32_t Bc = 4;
    constexpr int32_t threads = 2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    arm_align_alloc((void**)&Q_fp32_, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float), 16);
    arm_align_alloc((void**)&K_fp32_, B_ * S_K_ * KV_Head_ * D_ * sizeof(float), 16);
    arm_align_alloc((void**)&V_fp32_, B_ * S_K_ * KV_Head_ * D_ * sizeof(float), 16);

    for (int i = 0; i < B_ * S_Q_ * Q_Head_ * D_; ++i) Q_fp32_[i] = dist(gen);
    for (int i = 0; i < B_ * S_K_ * KV_Head_ * D_; ++i) {
      K_fp32_[i] = dist(gen);
      V_fp32_[i] = dist(gen);
    }

    arm_align_alloc((void**)&Q_h_, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float16_t), 16);
    arm_align_alloc((void**)&K_h_, B_ * S_K_ * KV_Head_ * D_ * sizeof(float16_t), 16);
    arm_align_alloc((void**)&V_h_, B_ * S_K_ * KV_Head_ * D_ * sizeof(float16_t), 16);

    for (int i = 0; i < B_ * S_Q_ * Q_Head_ * D_; ++i) Q_h_[i] = static_cast<float16_t>(Q_fp32_[i]);
    for (int i = 0; i < B_ * S_K_ * KV_Head_ * D_; ++i) {
      K_h_[i] = static_cast<float16_t>(K_fp32_[i]);
      V_h_[i] = static_cast<float16_t>(V_fp32_[i]);
    }

    arm_align_alloc((void**)&output_causal_, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float), 16);
    memset(output_causal_, 0, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float));

    arm_align_alloc((void**)&O_h_, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float16_t), 16);

    arm_align_alloc(&acc_s_cast_, threads * Br * Bc * sizeof(FlashAttnOp::dtype_t), 16);
    arm_align_alloc(&acc_o_, threads * Br * D_ * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&acc_s_, threads * Br * Bc * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&logsum_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&scoremax_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&scoremax_prev_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&score_scale_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
    arm_align_alloc(&score_sum_, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);

    arm_align_alloc((void**)&output_flash_, B_ * S_Q_ * Q_Head_ * D_ * sizeof(float), 16);

    fa_op_ = FlashAttnOp();
    fa_op_.init_workspace(
        (FlashAttnOp::dtype_t*)acc_s_cast_, (FlashAttnOp::acc_dtype_t*)acc_o_,
        (FlashAttnOp::acc_dtype_t*)acc_s_, (FlashAttnOp::acc_dtype_t*)logsum_,
        (FlashAttnOp::acc_dtype_t*)scoremax_, (FlashAttnOp::acc_dtype_t*)scoremax_prev_,
        (FlashAttnOp::acc_dtype_t*)score_scale_, (FlashAttnOp::acc_dtype_t*)score_sum_);
  }

  void CalculateRef() {
    causal_attention_gqa(output_causal_, Q_fp32_, K_fp32_, V_fp32_, B_, S_Q_, S_K_, Q_Head_,
                         KV_Head_, D_);
  }

  void Calculate() {
    fa_op_(Q_h_, K_h_, V_h_, O_h_, B_, Q_Head_, KV_Head_, S_Q_, S_K_, D_);
    for (int i = 0; i < B_ * S_Q_ * Q_Head_ * D_; ++i) { output_flash_[i] = (float)(O_h_[i]); }
  }

  bool Compare() {
    const float atol = 1e-2f;
    const float rtol = 1e-2f;
    bool all_close = true;
    float max_diff = 0.0f;

    for (int i = 0; i < B_ * S_Q_ * Q_Head_ * D_; ++i) {
      float diff = std::abs(output_causal_[i] - output_flash_[i]);
      float a = std::abs(output_causal_[i]);
      float b = std::abs(output_flash_[i]);

      if (diff > atol + rtol * std::max(a, b)) {
        all_close = false;
        max_diff = std::max(max_diff, diff);
      }
    }

    return all_close;
  }

  void TearDown() override {
    arm_align_free(Q_fp32_);
    arm_align_free(K_fp32_);
    arm_align_free(V_fp32_);
    arm_align_free(Q_h_);
    arm_align_free(K_h_);
    arm_align_free(V_h_);
    arm_align_free(output_causal_);
    arm_align_free(O_h_);
    arm_align_free(output_flash_);
    arm_align_free(acc_s_cast_);
    arm_align_free(acc_o_);
    arm_align_free(acc_s_);
    arm_align_free(logsum_);
    arm_align_free(scoremax_);
    arm_align_free(scoremax_prev_);
    arm_align_free(score_scale_);
    arm_align_free(score_sum_);
  }

 private:
  FlashAttnOp fa_op_;

  int B_;
  int S_Q_;
  int S_K_;
  int Q_Head_ = 16;
  int KV_Head_ = 8;
  int D_;

  float *Q_fp32_, *K_fp32_, *V_fp32_;
  float16_t *Q_h_, *K_h_, *V_h_, *O_h_;
  float* output_flash_;
  float* output_causal_;

  void* acc_s_cast_;
  void* acc_o_;
  void* acc_s_;
  void* logsum_;
  void* scoremax_;
  void* scoremax_prev_;
  void* score_scale_;
  void* score_sum_;
};

// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_1024_SK_1024_H_16_D_128) {
//   SetShape(1, 1024, 1024, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }

// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_1026_SK_1026_H_16_D_128) {
//   SetShape(1, 1026, 1026, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }

// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_512_SK_1024_H_16_D_128) {
//   SetShape(1, 512, 1024, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }

// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_127_SK_1025_H_16_D_128) {
//   SetShape(1, 127, 1025, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }
//
// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_1_SK_1024_H_16_D_128) {
//   SetShape(1, 1, 1024, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }

// TEST_F(ArmFlashAttn2_MHA_BSHD_FP16_Test, B_1_SQ_1_SK_1025_H_16_D_128) {
//   SetShape(1, 1, 1025, 16, 128);
//   AllocAndInitFACtx();
//   CalculateRef();
//   Calculate();
//   EXPECT_EQ(Compare(), true);
// }

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_1024_SK_1024_HQ_16_HKV_8_D_128) {
  SetShape(1, 1024, 1024, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_1026_SK_1026_HQ_16_HKV_8_D_128) {
  SetShape(1, 1026, 1026, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_512_SK_1024_HQ_16_HKV_8_D_128) {
  SetShape(1, 512, 1024, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_127_SK_1025_HQ_16_HKV_8_D_128) {
  SetShape(1, 127, 1025, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_1_SK_1024_HQ_16_HKV_8_D_128) {
  SetShape(1, 1, 1024, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(ArmFlashAttn2_GQA_BSHD_FP16_Test, B_1_SQ_1_SK_1025_HQ_16_HKV_8_D_128) {
  SetShape(1, 1, 1025, 128);
  AllocAndInitFACtx();
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
