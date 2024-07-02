#include "common.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>

__forceinline__ __device__ __host__ void sha256(hash_t& hash, const block_t& block)
{
    uint32_t a = aa;
    uint32_t b = bb;
    uint32_t c = cc;
    uint32_t d = dd;
    uint32_t e = ee;
    uint32_t f = ff;
    uint32_t g = gg;
    uint32_t h = hh;

    // finish the first 16 rounds using precalculated data
    sha256_round(m00, k[0], a, b, c, d, e, f, g, h);
    sha256_round(m01, k[1], a, b, c, d, e, f, g, h);
    sha256_round(m02, k[2], a, b, c, d, e, f, g, h);
    sha256_round(m03, k[3], a, b, c, d, e, f, g, h);
    sha256_round(m04, k[4], a, b, c, d, e, f, g, h);
    sha256_round(m05, k[5], a, b, c, d, e, f, g, h);
    sha256_round(m06, k[6], a, b, c, d, e, f, g, h);
    sha256_round(m07, k[7], a, b, c, d, e, f, g, h);
    sha256_round(m08, k[8], a, b, c, d, e, f, g, h);
    sha256_round(m09, k[9], a, b, c, d, e, f, g, h);
    sha256_round(m10, k[10], a, b, c, d, e, f, g, h);
    sha256_round(block.arr[11], k[11], a, b, c, d, e, f, g, h);
    sha256_round(block.arr[12], k[12], a, b, c, d, e, f, g, h);
    sha256_round(block.arr[13], k[13], a, b, c, d, e, f, g, h);
    sha256_round(m14, k[14], a, b, c, d, e, f, g, h);
    sha256_round(m15, k[15], a, b, c, d, e, f, g, h);

    uint32_t m16 = sha256_update_m_(m14, m09, m01, m00);
    uint32_t m17 = sha256_update_m_(m15, m10, m02, m01);
    uint32_t m18 = sha256_update_m_(m16, block.arr[11], m03, m02);
    uint32_t m19 = sha256_update_m_(m17, block.arr[12], m04, m03);
    uint32_t m20 = sha256_update_m_(m18, block.arr[13], m05, m04);
    uint32_t m21 = sha256_update_m_(m19, m14, m06, m05);
    uint32_t m22 = sha256_update_m_(m20, m15, m07, m06);
    uint32_t m23 = sha256_update_m_(m21, m16, m08, m07);
    uint32_t m24 = sha256_update_m_(m22, m17, m09, m08);
    uint32_t m25 = sha256_update_m_(m23, m18, m10, m09);
    uint32_t m26 = sha256_update_m_(m24, m19, block.arr[11], m10);
    uint32_t m27 = sha256_update_m_(m25, m20, block.arr[12], block.arr[11]);
    uint32_t m28 = sha256_update_m_(m26, m21, block.arr[13], block.arr[12]);
    uint32_t m29 = sha256_update_m_(m27, m22, m14, block.arr[13]);
    uint32_t m30 = sha256_update_m_(m28, m23, m15, m14);
    uint32_t m31 = sha256_update_m_(m29, m24, m16, m15);

    uint32_t m[16] = {
        m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31};

    // perform the remaining 48 rounds
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_round(m[i], k[16 + i], a, b, c, d, e, f, g, h);
    }
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_update_m(m, i);
        sha256_round(m[i], k[32 + i], a, b, c, d, e, f, g, h);
    }
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_update_m(m, i);
        sha256_round(m[i], k[48 + i], a, b, c, d, e, f, g, h);
    }

    hash.arr[0] = swap_endian(aa + a);
    hash.arr[1] = swap_endian(bb + b);
    hash.arr[2] = swap_endian(cc + c);
    hash.arr[3] = swap_endian(dd + d);
    hash.arr[4] = swap_endian(ee + e);
    hash.arr[5] = swap_endian(ff + f);
    hash.arr[6] = swap_endian(gg + g);
    hash.arr[7] = swap_endian(hh + h);
}

__forceinline__ __device__ __host__ bool less_than(const hash_t& lhs, const hash_t& rhs)
{
    DEVICE_UNROLL
    for (int i = 0; i < 8; ++i) {
        const uint32_t lhs_u32 = swap_endian(lhs.arr[i]);
        const uint32_t rhs_u32 = swap_endian(rhs.arr[i]);
        if (lhs_u32 < rhs_u32) {
            return true;
        } else if (rhs_u32 < lhs_u32) {
            return false;
        }
    }
    return false;
}

__forceinline__ __device__ uint8_t base64_to_ascii(int x)
{
    return x < 26 ? 65 + x : x < 52 ? 71 + x : x < 62 ? x - 4 : x < 63 ? 43 : 47;
}

constexpr int max_thread_count = 64 * 64 * 64 * 64;

/// \brief Encode a value in range [0, 64^4) to a u32 encoded as base64.
__forceinline__ __device__ uint32_t encode(int val)
{
    assert(0 <= val && val < max_thread_count);
    uint32_t ret{};
    for (int i = 0; i < 4; ++i) {
        ret |= base64_to_ascii(val % 64) << i * 8;
        val /= 64;
    }
    return ret;
}

__forceinline__ __device__ __host__ void copy(block_t& dst, const block_t& src)
{
    std::memcpy(dst.arr, src.arr, sizeof(dst.arr));
}

__forceinline__ __device__ __host__ void copy(hash_t& dst, const hash_t& src)
{
    std::memcpy(dst.arr, src.arr, sizeof(dst.arr));
}

__forceinline__ __device__ __host__ void set_worst_hash_value(hash_t& hash)
{
    std::memset(&hash, 0xff, sizeof(hash_t));
}

template <int block_size>
__global__ void __launch_bounds__(block_size) hash(int iteration, block_t* blocks)
{
    // initialize the block to a default state
    block_t block;
    set_common(block);

    // set the third to last u32 to the iteration number
    block.arr[11] = encode(iteration);

    // set the second to last u32 to the thread id
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    block.arr[12] = encode(idx);

    hash_t best_hash{};
    set_worst_hash_value(best_hash);
    block_t best_block{};

    // set the last u32 to the items handled by this thread
    for (int i = 0; i < 64; ++i) {
        const uint32_t mask_i = base64_to_ascii(i) << 24;
        for (int j = 0; j < 64; ++j) {
            const uint32_t mask_j = base64_to_ascii(j) << 16;
            for (int k = 0; k < 64; ++k) {
                const uint32_t mask_k = base64_to_ascii(k) << 8;
                block.arr[13]         = mask_i | mask_j | mask_k | uint32_t{0x80};

                hash_t hash;
                sha256(hash, block);

                if (less_than(hash, best_hash)) {
                    copy(best_hash, hash);
                    copy(best_block, block);
                }
            }
        }
    }

    struct reduction_type {
        hash_t hash;
        int i;
    } val;
    std::memcpy(&val.hash, &best_hash, sizeof(best_hash));
    val.i = threadIdx.x;

    using block_reduce = cub::BlockReduce<reduction_type, block_size>;
    __shared__ typename block_reduce::TempStorage tmp_storage;
    const reduction_type res =
        block_reduce(tmp_storage)
            .Reduce(
                val,
                [] __device__(const reduction_type& lhs, const reduction_type& rhs)
                    -> reduction_type { return less_than(lhs.hash, rhs.hash) ? lhs : rhs; });

    __shared__ int best_i;
    if (threadIdx.x == 0) {
        best_i = res.i;
    }
    __syncthreads();
    if (threadIdx.x == best_i) {
        copy(blocks[blockIdx.x], best_block);
    }
}

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA error at " __FILE__ ":%d \"%s\"\n", __LINE__, cudaGetErrorString(err));   \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

void print_input(const block_t& block)
{
    for (int i = 0; i < 13; ++i) {
        const uint32_t tmp = swap_endian(block.arr[i]);
        for (int j = 0; j < 4; ++j) {
            printf("%c", reinterpret_cast<const char*>(&tmp)[j]);
        }
    }
    const uint32_t tmp = swap_endian(block.arr[13]);
    for (int j = 0; j < 3; ++j) {
        printf("%c", reinterpret_cast<const char*>(&tmp)[j]);
    }
    printf("\n");
}

void print_hash(const hash_t& hash)
{
    for (int i = 0; i < 8; ++i) {
        printf("%08x ", swap_endian(hash.arr[i]));
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    int iter_offset = 0;
    if (argc > 1) {
        iter_offset = std::strtol(argv[1], nullptr, 10);
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    constexpr int grid_size  = 256;
    constexpr int block_size = 256;
    static_assert(grid_size * block_size <= max_thread_count);

    block_t* d_blocks{};
    CHECK_CUDA(cudaMalloc(&d_blocks, grid_size * sizeof(block_t)));

    block_t best_block{};
    hash_t best_hash;
    set_worst_hash_value(best_hash);

    const int num_batches = 100;
    for (int i = 0; i < num_batches; ++i) {
        // process in batches to reduce synchronization overhead
        CHECK_CUDA(cudaEventRecord(start, stream));

        const int num_iters_per_batch = 2;
        for (int j = 0; j < num_iters_per_batch; ++j) {
            const int iteration = num_iters_per_batch * i + j;
            hash<block_size><<<grid_size, block_size, 0 /* shared memory */, stream>>>(
                iter_offset + iteration, d_blocks);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));

        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds{};
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        const double hash_count =
            static_cast<double>(num_iters_per_batch) * grid_size * block_size * 64 * 64 * 64;
        const double seconds = milliseconds / 1000.;
        printf(
            "iter [%d, %d): %fGH/s (%fms)\n",
            iter_offset + num_iters_per_batch * i,
            iter_offset + num_iters_per_batch * (i + 1),
            hash_count / seconds / 1.e9,
            milliseconds);

        std::vector<block_t> h_blocks(grid_size);
        CHECK_CUDA(cudaMemcpy(
            h_blocks.data(), d_blocks, grid_size * sizeof(block_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < grid_size; ++i) {
            hash_t hash{};
            sha256(hash, h_blocks[i]);
            if (less_than(hash, best_hash)) {
                copy(best_block, h_blocks[i]);
                copy(best_hash, hash);
                print_input(best_block);
                print_hash(best_hash);
            }
        }
    }

    printf("final result:\n");
    print_input(best_block);
    print_hash(best_hash);

    CHECK_CUDA(cudaFree(d_blocks));

    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));

    CHECK_CUDA(cudaStreamDestroy(stream));
}
