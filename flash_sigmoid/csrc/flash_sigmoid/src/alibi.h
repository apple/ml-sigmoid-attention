#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal>
struct Alibi {

    const float alibi_slope;
    const int max_seqlen_diff;

    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_diff(max_seqlen_k - max_seqlen_q)
        {};


    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

        /*--------------------------------------------------------------------------------
        Unlike FlashAttention, FlashSigmoid requires computing full ALiBi offsets.
        This is because sigmoid is NOT translation invariant.
        ````
        offset = row_idx + max_seqlen_k - max_seqlen_q - col_idx
        = ((row_idx_offset + mi * warp_row_stride) + i * 8)
            + (max_seqlen_k - max_seqlen_q)
            - (((col_idx_offset_ + (lane_id % 4) * 2) + nj * 8) + j)
        ```
        We save computations by refactoring common addendums out of the loop.
        --------------------------------------------------------------------------------*/
        const float base_off = max_seqlen_diff + row_idx_offset - col_idx_offset;

        #pragma unroll
        for (int mi = 0, base_offset=0; mi < size<0, 1>(tensor); ++mi, base_offset += warp_row_stride) {
            #pragma unroll
            for (int i = 0, row_offset = base_offset; i < size<0, 0>(tensor); ++i, row_offset += 8) {
                #pragma unroll
                for (int nj = 0, col_1_offset = row_offset; nj < size<1, 1>(tensor); ++nj, col_1_offset -= 8) {
                    #pragma unroll
                    for (int j = 0, total_offset = col_1_offset; j < size<1, 0>(tensor); ++j, --total_offset) {
                        if constexpr (Is_causal) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * (base_off + total_offset);
                        } else {
                            tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(base_off + total_offset);
                        }
                    }
                }
            }
        }
    }

};

}  // namespace flash
