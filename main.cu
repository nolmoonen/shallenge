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
    uint32_t a = 0xd6d54205;
    uint32_t b = 0x06cc1c4a;
    uint32_t c = 0xf2df215f;
    uint32_t d = 0x106b42e3;
    uint32_t e = 0xb2368541;
    uint32_t f = 0x1b7e592a;
    uint32_t g = 0x110dae68;
    uint32_t h = 0x6ed0d95b;

    uint32_t m[16];
    std::memcpy(m, block.arr, sizeof(m));

    // finish the first 16 rounds using precalculated data
    DEVICE_UNROLL
    for (int i = 10; i < 16; ++i) {
        sha256_round(m, a, b, c, d, e, f, g, h, 0, i);
    }
    // perform the remaining 48 rounds
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_update_m(m, i);
        sha256_round(m, a, b, c, d, e, f, g, h, 16, i);
    }
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_update_m(m, i);
        sha256_round(m, a, b, c, d, e, f, g, h, 32, i);
    }
    DEVICE_UNROLL
    for (int i = 0; i < 16; ++i) {
        sha256_update_m(m, i);
        sha256_round(m, a, b, c, d, e, f, g, h, 48, i);
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
    block.arr[num_inputs_u32 - 3] = encode(iteration);

    // set the second to last u32 to the thread id
    const int idx                 = blockDim.x * blockIdx.x + threadIdx.x;
    block.arr[num_inputs_u32 - 2] = encode(idx);

    hash_t best_hash{};
    set_worst_hash_value(best_hash);
    block_t best_block{};

    // set the last u32 to the items handled by this thread
    constexpr int ascii_lowercase_a = 97;
    for (int i = 0; i < 26; ++i) {
        const uint32_t mask_i = (ascii_lowercase_a + i) << 24;
        for (int j = 0; j < 26; ++j) {
            const uint32_t mask_j = (ascii_lowercase_a + j) << 16;
            for (int k = 0; k < 26; ++k) {
                const uint32_t mask_k = (ascii_lowercase_a + k) << 8;
                for (int l = 0; l < 26; ++l) {
                    const uint32_t mask_l         = ascii_lowercase_a + l;
                    block.arr[num_inputs_u32 - 1] = mask_i | mask_j | mask_k | mask_l;

                    hash_t hash;
                    sha256(hash, block);

                    if (less_than(hash, best_hash)) {
                        copy(best_hash, hash);
                        copy(best_block, block);
                    }
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
    for (int i = 0; i < num_inputs_u32; ++i) {
        const uint32_t tmp = swap_endian(block.arr[i]);
        for (int j = 0; j < 4; ++j) {
            printf("%c", reinterpret_cast<const char*>(&tmp)[j]);
        }
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
            static_cast<double>(num_iters_per_batch) * grid_size * block_size * 26 * 26 * 26 * 26;
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
