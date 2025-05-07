#include <benchmark/benchmark.h>
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

static void BSHD_FP16_GQA(benchmark::State& state) {
  int32_t S_QKV = static_cast<int32_t>(state.range(0));

  constexpr int32_t B = 1;
  int32_t S_Q = S_QKV;
  int32_t S_K = S_QKV;
  constexpr int32_t Q_Head = 32;
  constexpr int32_t KV_Head = 8;
  constexpr int32_t D = 64;

  constexpr int32_t Br = 4;
  constexpr int32_t Bc = 4;
  constexpr int32_t threads = 4;
  constexpr bool high_precession_exp = false;

  using FlashAttnOp = FlashAttn2<NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<
      Br, Bc, Q_Head, KV_Head, threads, high_precession_exp>>;

  FlashAttnOp::dtype_t *Q_h, *K_h, *V_h, *O_h;
  void* acc_s_cast;
  void* acc_o;
  void* acc_s;
  void* logsum;
  void* scoremax;
  void* scoremax_prev;
  void* score_scale;
  void* score_sum;

  arm_align_alloc((void**)&Q_h, B * S_Q * Q_Head * D * sizeof(FlashAttnOp::dtype_t), 16);
  arm_align_alloc((void**)&K_h, B * S_K * KV_Head * D * sizeof(FlashAttnOp::dtype_t), 16);
  arm_align_alloc((void**)&V_h, B * S_K * KV_Head * D * sizeof(FlashAttnOp::dtype_t), 16);
  arm_align_alloc((void**)&O_h, B * S_Q * Q_Head * D * sizeof(FlashAttnOp::dtype_t), 16);
  arm_align_alloc(&acc_s_cast, threads * Br * Bc * sizeof(FlashAttnOp::dtype_t), 16);
  arm_align_alloc(&acc_o, threads * Br * D * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&acc_s, threads * Br * Bc * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&logsum, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&scoremax, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&scoremax_prev, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&score_scale, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);
  arm_align_alloc(&score_sum, threads * Br * sizeof(FlashAttnOp::acc_dtype_t), 16);

  auto fa_op = FlashAttnOp();
  fa_op.init_workspace(
      (FlashAttnOp::dtype_t*)acc_s_cast, (FlashAttnOp::acc_dtype_t*)acc_o,
      (FlashAttnOp::acc_dtype_t*)acc_s, (FlashAttnOp::acc_dtype_t*)logsum,
      (FlashAttnOp::acc_dtype_t*)scoremax, (FlashAttnOp::acc_dtype_t*)scoremax_prev,
      (FlashAttnOp::acc_dtype_t*)score_scale, (FlashAttnOp::acc_dtype_t*)score_sum);

  for (auto _ : state) { fa_op(Q_h, K_h, V_h, O_h, B, Q_Head, S_Q, S_K, D, true); }

  arm_align_free(Q_h);
  arm_align_free(K_h);
  arm_align_free(V_h);
  arm_align_free(O_h);
  arm_align_free(acc_s_cast);
  arm_align_free(acc_o);
  arm_align_free(acc_s);
  arm_align_free(logsum);
  arm_align_free(scoremax);
  arm_align_free(scoremax_prev);
  arm_align_free(score_scale);
  arm_align_free(score_sum);
}

BENCHMARK(BSHD_FP16_GQA)->RangeMultiplier(2)->Range(32, 2048);
BENCHMARK_MAIN();
