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

namespace mobi_attn {
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
                         const int32_t batch_size, const int32_t q_head_size,
                         const int32_t kv_head_size, const int32_t seq_size_q,
                         const int32_t seq_size_k, const int32_t dim_size) {
    __impl().fa2(Q, K, V, O, batch_size, q_head_size, kv_head_size, seq_size_q, seq_size_k,
                 dim_size);
  }

 private:
  FlashAttn2Impl fa2_impl_;
};
}  // namespace mobi_attn
