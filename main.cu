#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
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

constexpr uint32_t aa = 0x6a09e667;
constexpr uint32_t bb = 0xbb67ae85;
constexpr uint32_t cc = 0x3c6ef372;
constexpr uint32_t dd = 0xa54ff53a;
constexpr uint32_t ee = 0x510e527f;
constexpr uint32_t ff = 0x9b05688c;
constexpr uint32_t gg = 0x1f83d9ab;
constexpr uint32_t hh = 0x5be0cd19;

struct nonce_t {
    uint32_t m11;
    uint32_t m12;
    uint32_t m13;
};

struct hash_t {
    uint32_t arr[8];
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
    // return (x & y) ^ (~x & z);
    // https://github.com/hashcat/hashcat/blob/master/OpenCL/inc_hash_sha256.h
    return z ^ (x & (y ^ z));
}
__forceinline__ __device__ __host__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    // return (x & y) ^ (x & z) ^ (y & z);
    // https://github.com/hashcat/hashcat/blob/master/OpenCL/inc_hash_sha256.h
    return (x & y) | (z & (x ^ y));
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

__forceinline__ __device__ __host__ void sha256_round(
    uint32_t m,
    uint32_t k,
    uint32_t& a,
    uint32_t& b,
    uint32_t& c,
    uint32_t& d,
    uint32_t& e,
    uint32_t& f,
    uint32_t& g,
    uint32_t& h)
{
    const uint32_t t1 = h + ep1(e) + ch(e, f, g) + k + m;
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

__forceinline__ __device__ __host__ uint32_t
sha256_update_m_(uint32_t i2, uint32_t i7, uint32_t i15, uint32_t i16)
{
    return sig1(i2) + i7 + sig0(i15) + i16;
}

__forceinline__ __device__ __host__ void sha256_update_m(uint32_t (&m)[16], int i)
{
    m[i] = sha256_update_m_(
        m[(i + 16 - 2) % 16], m[(i + 16 - 7) % 16], m[(i + 16 - 15) % 16], m[(i + 16 - 16) % 16]);
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

constexpr uint32_t m00 = 0x6e6f6c2f; // 'nol/'
constexpr uint32_t m01 = 0x30303030; // '0000'
constexpr uint32_t m02 = 0x30303030;
constexpr uint32_t m03 = 0x30303030;
constexpr uint32_t m04 = 0x30303030;
constexpr uint32_t m05 = 0x30303030;
constexpr uint32_t m06 = 0x30303030;
constexpr uint32_t m07 = 0x30303030;
constexpr uint32_t m08 = 0x30303030;
constexpr uint32_t m09 = 0x30303030;
constexpr uint32_t m10 = 0x30303030;
// m11 is iteration
// m12 is thread index
// m13 is thread variable and single bit padding
constexpr uint32_t m14 = 0x00000000; // upper part of u64 size
constexpr uint32_t m15 = 0x000001b8; // length, 64 - 8 - 1 = 55 * 8 = 440 in u32 big endian

__forceinline__ __device__ __host__ void sha256(
    hash_t& hash, uint32_t m11, uint32_t m12, uint32_t m13)
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
    sha256_round(m11, k[11], a, b, c, d, e, f, g, h);
    sha256_round(m12, k[12], a, b, c, d, e, f, g, h);
    sha256_round(m13, k[13], a, b, c, d, e, f, g, h);
    sha256_round(m14, k[14], a, b, c, d, e, f, g, h);
    sha256_round(m15, k[15], a, b, c, d, e, f, g, h);

    uint32_t m16 = sha256_update_m_(m14, m09, m01, m00);
    uint32_t m17 = sha256_update_m_(m15, m10, m02, m01);
    uint32_t m18 = sha256_update_m_(m16, m11, m03, m02);
    uint32_t m19 = sha256_update_m_(m17, m12, m04, m03);
    uint32_t m20 = sha256_update_m_(m18, m13, m05, m04);
    uint32_t m21 = sha256_update_m_(m19, m14, m06, m05);
    uint32_t m22 = sha256_update_m_(m20, m15, m07, m06);
    uint32_t m23 = sha256_update_m_(m21, m16, m08, m07);
    uint32_t m24 = sha256_update_m_(m22, m17, m09, m08);
    uint32_t m25 = sha256_update_m_(m23, m18, m10, m09);
    uint32_t m26 = sha256_update_m_(m24, m19, m11, m10);
    uint32_t m27 = sha256_update_m_(m25, m20, m12, m11);
    uint32_t m28 = sha256_update_m_(m26, m21, m13, m12);
    uint32_t m29 = sha256_update_m_(m27, m22, m14, m13);
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

    hash.arr[0] = aa + a;
    hash.arr[1] = bb + b;
    hash.arr[2] = cc + c;
    hash.arr[3] = dd + d;
    hash.arr[4] = ee + e;
    hash.arr[5] = ff + f;
    hash.arr[6] = gg + g;
    hash.arr[7] = hh + h;
}

__forceinline__ __device__ __host__ bool less_than(const hash_t& lhs, const hash_t& rhs)
{
    DEVICE_UNROLL
    for (int i = 0; i < 8; ++i) {
        if (lhs.arr[i] < rhs.arr[i]) {
            return true;
        } else if (rhs.arr[i] < lhs.arr[i]) {
            return false;
        }
    }
    return false;
}

constexpr int base64_max = 62;

__forceinline__ __device__ uint8_t base64_to_ascii(int x)
{
    assert(0 <= x && x < 62);
    __builtin_assume(0 <= x && x < 62);
    return x < 26 ? 65 + x : x < 52 ? 71 + x : x - 4;
}

constexpr int max_thread_count = base64_max * base64_max * base64_max * base64_max;

/// \brief Encode a value in range [0, base64_max^4) to a u32 encoded as base64.
__forceinline__ __device__ uint32_t encode(int val)
{
    assert(0 <= val && val < max_thread_count);
    uint32_t ret{};
    for (int i = 0; i < 4; ++i) {
        ret |= base64_to_ascii(val % base64_max) << i * 8;
        val /= base64_max;
    }
    return ret;
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
__global__ void __launch_bounds__(block_size) hash(int iteration, nonce_t* nonces)
{
    // set the third to last u32 to the iteration number
    const uint32_t m11 = encode(iteration);

    // set the second to last u32 to the thread id
    const int idx      = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t m12 = encode(idx);

    hash_t best_hash{};
    set_worst_hash_value(best_hash);
    uint32_t best_m13{};

    // set the last u32 to the items handled by this thread
    for (int i = 0; i < base64_max; ++i) {
        const uint32_t mask_i = base64_to_ascii(i) << 24;
        for (int j = 0; j < base64_max; ++j) {
            const uint32_t mask_j = base64_to_ascii(j) << 16;
            for (int k = 0; k < base64_max; ++k) {
                const uint32_t mask_k = base64_to_ascii(k) << 8;
                const uint32_t m13    = mask_i | mask_j | mask_k | uint32_t{0x80};

                hash_t hash;
                sha256(hash, m11, m12, m13);

                if (less_than(hash, best_hash)) {
                    copy(best_hash, hash);
                    best_m13 = m13;
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
        nonces[blockIdx.x].m11 = m11;
        nonces[blockIdx.x].m12 = m12;
        nonces[blockIdx.x].m13 = best_m13;
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

void print_u32_as_char(uint32_t x, int n = 4)
{
    const uint32_t tmp = swap_endian(x);
    for (int j = 0; j < n; ++j) {
        printf("%c", reinterpret_cast<const char*>(&tmp)[j]);
    }
}

void print_input(const nonce_t& nonce)
{
    print_u32_as_char(m00);
    print_u32_as_char(m01);
    print_u32_as_char(m02);
    print_u32_as_char(m03);
    print_u32_as_char(m04);
    print_u32_as_char(m05);
    print_u32_as_char(m06);
    print_u32_as_char(m07);
    print_u32_as_char(m08);
    print_u32_as_char(m09);
    print_u32_as_char(m10);
    print_u32_as_char(nonce.m11);
    print_u32_as_char(nonce.m12);
    print_u32_as_char(nonce.m13, 3);
    printf("\n");
}

void print_hash(const hash_t& hash)
{
    for (int i = 0; i < 8; ++i) {
        printf("%08x ", hash.arr[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    setbuf(stdout, nullptr); // make stream unbuffered

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

    nonce_t* d_nonces{};
    CHECK_CUDA(cudaMalloc(&d_nonces, grid_size * sizeof(nonce_t)));

    nonce_t best_nonce{};
    hash_t best_hash;
    set_worst_hash_value(best_hash);

    const int num_batches = INT_MAX;
    for (int i = 0; i < num_batches; ++i) {
        // process in batches to reduce synchronization overhead
        CHECK_CUDA(cudaEventRecord(start, stream));

        const int num_iters_per_batch = 2;
        for (int j = 0; j < num_iters_per_batch; ++j) {
            const int iteration = num_iters_per_batch * i + j;
            hash<block_size><<<grid_size, block_size, 0 /* shared memory */, stream>>>(
                iter_offset + iteration, d_nonces);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));

        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds{};
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        const double hash_count = static_cast<double>(num_iters_per_batch) * grid_size *
                                  block_size * base64_max * base64_max * base64_max;
        const double seconds = milliseconds / 1000.;
        printf(
            "iter [%d, %d): %fGH/s (%fms)\n",
            iter_offset + num_iters_per_batch * i,
            iter_offset + num_iters_per_batch * (i + 1),
            hash_count / seconds / 1.e9,
            milliseconds);

        std::vector<nonce_t> h_nonces(grid_size);
        CHECK_CUDA(cudaMemcpy(
            h_nonces.data(), d_nonces, grid_size * sizeof(nonce_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < grid_size; ++i) {
            hash_t hash{};
            sha256(hash, h_nonces[i].m11, h_nonces[i].m12, h_nonces[i].m13);
            if (less_than(hash, best_hash)) {
                best_nonce = h_nonces[i];
                copy(best_hash, hash);
                print_input(best_nonce);
                print_hash(best_hash);
            }
        }
    }

    printf("final result:\n");
    print_input(best_nonce);
    print_hash(best_hash);

    CHECK_CUDA(cudaFree(d_nonces));

    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));

    CHECK_CUDA(cudaStreamDestroy(stream));
}
