#pragma once

#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>
#include <cmath>

namespace mobi_attn {

static inline float32x4_t vclampq_f32(float32x4_t x, float lo, float hi) {
  x = vminq_f32(x, vdupq_n_f32(hi));
  x = vmaxq_f32(x, vdupq_n_f32(lo));
  return x;
};

// NOTE: Slow.
static inline float32x4_t vexpq_hp_f32(float32x4_t x) {
  float result[4];
  vst1q_f32(result, x);

#pragma unroll
  for (float& i : result) { i = expf(i); }

  return vld1q_f32(result);
}

// ref from ncnn.
//
// NEON implementation of exp
//
// Inspired by Intel Approximate Math library, and based on the
// corresponding algorithms of the cephes math library
//
// see:
// https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun.h#L123
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f
#define c_cephes_LOG2EF 1.44269504088896341

#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

static inline float32x4_t vexpq_fast_f32(float32x4_t x) {
  float32x4_t tmp, fx;

  float32x4_t one = vdupq_n_f32(1);
  x = vclampq_f32(x, c_exp_lo, c_exp_hi);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  z = vmulq_f32(x, x);

  float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);

  y = vmlaq_f32(x, y, z);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

}  // namespace mobi_attn

#endif