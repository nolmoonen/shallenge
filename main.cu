#include <cub/block/block_reduce.cuh>

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifdef __CUDA_ARCH__
#define DEVICE_UNROLL #pragma unroll
#else
#define DEVICE_UNROLL
#endif

// no need to optimize loading, these constants are inlined
#ifdef __CUDA_ARCH__
__device__
#endif
    constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};

constexpr int block_size_u32 = 512 / 32;
constexpr int hash_size_u32  = 256 / 32;
constexpr int block_size_u8  = 512 / 8;
constexpr int num_inputs_u8  = block_size_u8 - 8 - 4; // len in u64 and bit padding in u32
constexpr int num_inputs_u32 = num_inputs_u8 / 4;

struct block_t {
    uint32_t arr[block_size_u32];
};

struct hash_t {
    uint32_t arr[hash_size_u32];
};

__forceinline__ __device__ __host__ uint32_t rotr(uint32_t a, int b)
{
#if __CUDA_ARCH__ and 0
    // seems to be worse, looks like the compiler can already figure this out
    return __funnelshift_r(a, a, b);
#else
    return (a >> b) | (a << (32 - b));
#endif
}
__forceinline__ __device__ __host__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}
__forceinline__ __device__ __host__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}
__forceinline__ __device__ __host__ uint32_t ep0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
__forceinline__ __device__ __host__ uint32_t ep1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
__forceinline__ __device__ __host__ uint32_t sig0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
__forceinline__ __device__ __host__ uint32_t sig1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__forceinline__ __device__ __host__ uint32_t swap_endian(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __byte_perm(x, uint32_t{0}, uint32_t{0x0123});
#else
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&x);
    return uint32_t{ptr[3]} | (uint32_t{ptr[2]} << 8) | (uint32_t{ptr[1]} << 16) |
           (uint32_t{ptr[0]} << 24);
#endif
}

__forceinline__ __device__ __host__ void sha256(hash_t& hash, const block_t& block)
{
    uint32_t m[16];
    std::memcpy(m, block.arr, sizeof(block.arr));

    constexpr uint32_t aa = 0x6a09e667;
    constexpr uint32_t bb = 0xbb67ae85;
    constexpr uint32_t cc = 0x3c6ef372;
    constexpr uint32_t dd = 0xa54ff53a;
    constexpr uint32_t ee = 0x510e527f;
    constexpr uint32_t ff = 0x9b05688c;
    constexpr uint32_t gg = 0x1f83d9ab;
    constexpr uint32_t hh = 0x5be0cd19;

    uint32_t a = aa;
    uint32_t b = bb;
    uint32_t c = cc;
    uint32_t d = dd;
    uint32_t e = ee;
    uint32_t f = ff;
    uint32_t g = gg;
    uint32_t h = hh;

    DEVICE_UNROLL
    for (int j = 0; j < 4; j++) {
        DEVICE_UNROLL
        for (int i = 0; i < 16; ++i) {
            const uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[16 * j + i] + m[i];
            const uint32_t t2 = ep0(a) + maj(a, b, c);

            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        if (j < 3) {
            DEVICE_UNROLL
            for (int i = 0; i < 16; ++i) {
                m[i] = sig1(m[(i + 16 - 2) % 16]) + m[(i + 16 - 7) % 16] +
                       sig0(m[(i + 16 - 15) % 16]) + m[(i + 16 - 16) % 16];
            }
        }
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
    for (int i = 0; i < 4; ++i) {
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
__global__ void hash(int iteration, block_t* blocks)
{
    // initialize the block to a default state
    block_t block;
    block.arr[0] =
        uint32_t{'n'} << 24 | (uint32_t{'o'} << 16) | (uint32_t{'l'} << 8) | (uint32_t{'/'});
    for (int i = 0; i < num_inputs_u32; ++i) {
        block.arr[1 + i] =
            uint32_t{'0'} | (uint32_t{'0'} << 8) | (uint32_t{'0'} << 16) | (uint32_t{'0'} << 24);
    }
    block.arr[block_size_u32 - 3] = swap_endian(uint32_t{0x80}); // single bit padding
    block.arr[block_size_u32 - 2] = 0;
    // length, 64 - 8 - 4 = 52 * 8 = 416 in u32 big endian
    block.arr[block_size_u32 - 1] = uint32_t{416};

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

int main()
{
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
            hash<block_size>
                <<<grid_size, block_size, 0 /* shared memory */, stream>>>(iteration, d_blocks);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));

        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds{};
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        const double hash_count =
            static_cast<double>(num_iters_per_batch) * grid_size * block_size * 26 * 26 * 26 * 26;
        const double seconds = milliseconds / 1000.;
        printf("%fGH/s (%fms)\n", hash_count / seconds / 1.e9, milliseconds);

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
