/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////


/*--------------------------------------------------------------------------------
Compute and return `tanh` of given scalar `a`.

Solution credits--
1. Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis
https://arxiv.org/pdf/2208.11174

2. https://forums.developer.nvidia.com/t/hardware-accelerated-computation-of-the-sigmoid-logistic-function/266206
Njuffa: contributor at NVIDIA blogs who mentions `tanh.approx.f32`.
--------------------------------------------------------------------------------*/
__forceinline__ __device__ float fast_tanhf (float a)
{
    float r;
    asm ("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(a));
    return r;
}


/*--------------------------------------------------------------------------------
Compute element-wise sigmoid of given tensor after scaling it.

:param tensor: The tensor whose sigmoid needs to be computed.
:param scale: The scale with which to multiply the tensor.

:returns: Element-wise sigmoid `sigmoid(scale * tensor)`.

Notes--
1. The relation between `softmax` and `sigmoid` is:
`sigmoid(x) = 0.5*(1 + tanh(0.5*x))`. However, in our code, we take the
factor of `0.5` from `0.5*x` and incorporate it in `scale` variable by
pre-multiplying it.
--------------------------------------------------------------------------------*/
template <typename Engine0, typename Layout0>
__forceinline__ __device__
void apply_sigmoid(Tensor<Engine0, Layout0> &tensor, const float scale) {
    #pragma unroll
    for (int mi = 0; mi < size(tensor); ++mi) {
        tensor(mi) = fmaf(0.5, fast_tanhf(tensor(mi) * scale), 0.5f);
    }
}


/*--------------------------------------------------------------------------------
Define per-location application of backprop of sigmoid attention.

:param a: The value of a pixel of sigmoid attention `p`.
:param b: The corrected `dp` value based on the check in the code.

:returns: A possibly faster answer of `a*(1 - a)*b`, which is the gradient
    backpropagated through sigmoid attention mechanism.

Notes--
1. The intuition is that `a*(1 - a)*b = a*b - a^2 * b = fmaf(-a, a*b, a*b)`.
Thus, maybe we can compute the answer faster with 2 `fmaf` applications as:
```
float ab = fmaf(a, b, 0);
return fmaf(-a, ab, ab);
```
--------------------------------------------------------------------------------*/
__forceinline__ __device__
float fmaf_sigmoid_backprop(float a, float b) {
    float ab = a * b;
    return fmaf(-a, ab, ab);
}


/*--------------------------------------------------------------------------------
Define a possibly more efficient implementation of passing gradients back
from the outputs of the attention mechanism to its inputs.

:param p: The tensor of output of sigmoid attention activations.
:param dp: The tensor of gradient of loss with respect to sigmoid attention activations.

:returns: Nothing. The `dp` is updated inplace with the answer.
--------------------------------------------------------------------------------*/
template <bool Is_dropout, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__
void apply_sigmoid_backprop(
    Tensor<Engine0, Layout0> &p,
    Tensor<Engine1, Layout1> &dp
) {
    // Unroll for each row.
    #pragma unroll
    for (int mi = 0; mi < size<0>(dp); ++mi) {
        // Unroll for each column.
        # pragma unroll
        for (int ni = 0; ni < size<1>(dp); ++ni) {
            // Compute the tri-conditional.
            const float a = p(mi, ni);
            const float b = dp(mi, ni);
            const float corrected_b = !Is_dropout || a >= 0 ? b : 0.f;

            // Compute and fill the answer in the second input tensor.
            dp(mi, ni) = fmaf_sigmoid_backprop(
                /*a=*/a,
                /*b=*/corrected_b
            );
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    __forceinline__ __device__ Softmax() {};

    template<typename Tensor0>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        apply_sigmoid(scores, softmax_scale_log2);
    };
};

}  // namespace flash
