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

#define SHA256_ROTR(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define SHA256_MAJ(x, y, z) (((x) & (y)) | ((z) & ((x) ^ (y))))
#define SHA256_CH(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))

#define SHA256_EP0(x) (SHA256_ROTR(x, 2) ^ SHA256_ROTR(x, 13) ^ SHA256_ROTR(x, 22))
#define SHA256_EP1(x) (SHA256_ROTR(x, 6) ^ SHA256_ROTR(x, 11) ^ SHA256_ROTR(x, 25))
#define SHA256_SIG0(x) (SHA256_ROTR(x, 7) ^ SHA256_ROTR(x, 18) ^ (x >> 3))
#define SHA256_SIG1(x) (SHA256_ROTR(x, 17) ^ SHA256_ROTR(x, 19) ^ (x >> 10))

#define SHA256_STEP(m, k, a, b, c, d, e, f, g, h)                                                  \
    h += SHA256_EP1(e) + SHA256_CH(e, f, g) + k + m;                                               \
    d += h;                                                                                        \
    h += SHA256_EP0(a) + SHA256_MAJ(a, b, c);

#define SHA256_EXPAND(i2, i7, i15, i16) (SHA256_SIG1(i2) + i7 + SHA256_SIG0(i15) + i16)

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

    SHA256_STEP(m00, k[0x00], a, b, c, d, e, f, g, h);
    SHA256_STEP(m01, k[0x01], h, a, b, c, d, e, f, g);
    SHA256_STEP(m02, k[0x02], g, h, a, b, c, d, e, f);
    SHA256_STEP(m03, k[0x03], f, g, h, a, b, c, d, e);
    SHA256_STEP(m04, k[0x04], e, f, g, h, a, b, c, d);
    SHA256_STEP(m05, k[0x05], d, e, f, g, h, a, b, c);
    SHA256_STEP(m06, k[0x06], c, d, e, f, g, h, a, b);
    SHA256_STEP(m07, k[0x07], b, c, d, e, f, g, h, a);
    SHA256_STEP(m08, k[0x08], a, b, c, d, e, f, g, h);
    SHA256_STEP(m09, k[0x09], h, a, b, c, d, e, f, g);
    SHA256_STEP(m10, k[0x0a], g, h, a, b, c, d, e, f);
    SHA256_STEP(m11, k[0x0b], f, g, h, a, b, c, d, e);
    SHA256_STEP(m12, k[0x0c], e, f, g, h, a, b, c, d);
    SHA256_STEP(m13, k[0x0d], d, e, f, g, h, a, b, c);
    SHA256_STEP(m14, k[0x0e], c, d, e, f, g, h, a, b);
    SHA256_STEP(m15, k[0x0f], b, c, d, e, f, g, h, a);

    uint32_t m0 = m00;
    uint32_t m1 = m01;
    uint32_t m2 = m02;
    uint32_t m3 = m03;
    uint32_t m4 = m04;
    uint32_t m5 = m05;
    uint32_t m6 = m06;
    uint32_t m7 = m07;
    uint32_t m8 = m08;
    uint32_t m9 = m09;
    uint32_t ma = m10;
    uint32_t mb = m11;
    uint32_t mc = m12;
    uint32_t md = m13;
    uint32_t me = m14;
    uint32_t mf = m15;

    m0 = SHA256_EXPAND(me, m9, m1, m0);
    m1 = SHA256_EXPAND(mf, ma, m2, m1);
    m2 = SHA256_EXPAND(m0, mb, m3, m2);
    m3 = SHA256_EXPAND(m1, mc, m4, m3);
    m4 = SHA256_EXPAND(m2, md, m5, m4);
    m5 = SHA256_EXPAND(m3, me, m6, m5);
    m6 = SHA256_EXPAND(m4, mf, m7, m6);
    m7 = SHA256_EXPAND(m5, m0, m8, m7);
    m8 = SHA256_EXPAND(m6, m1, m9, m8);
    m9 = SHA256_EXPAND(m7, m2, ma, m9);
    ma = SHA256_EXPAND(m8, m3, mb, ma);
    mb = SHA256_EXPAND(m9, m4, mc, mb);
    mc = SHA256_EXPAND(ma, m5, md, mc);
    md = SHA256_EXPAND(mb, m6, me, md);
    me = SHA256_EXPAND(mc, m7, mf, me);
    mf = SHA256_EXPAND(md, m8, m0, mf);

    SHA256_STEP(m0, k[0x10], a, b, c, d, e, f, g, h);
    SHA256_STEP(m1, k[0x11], h, a, b, c, d, e, f, g);
    SHA256_STEP(m2, k[0x12], g, h, a, b, c, d, e, f);
    SHA256_STEP(m3, k[0x13], f, g, h, a, b, c, d, e);
    SHA256_STEP(m4, k[0x14], e, f, g, h, a, b, c, d);
    SHA256_STEP(m5, k[0x15], d, e, f, g, h, a, b, c);
    SHA256_STEP(m6, k[0x16], c, d, e, f, g, h, a, b);
    SHA256_STEP(m7, k[0x17], b, c, d, e, f, g, h, a);
    SHA256_STEP(m8, k[0x18], a, b, c, d, e, f, g, h);
    SHA256_STEP(m9, k[0x19], h, a, b, c, d, e, f, g);
    SHA256_STEP(ma, k[0x1a], g, h, a, b, c, d, e, f);
    SHA256_STEP(mb, k[0x1b], f, g, h, a, b, c, d, e);
    SHA256_STEP(mc, k[0x1c], e, f, g, h, a, b, c, d);
    SHA256_STEP(md, k[0x1d], d, e, f, g, h, a, b, c);
    SHA256_STEP(me, k[0x1e], c, d, e, f, g, h, a, b);
    SHA256_STEP(mf, k[0x1f], b, c, d, e, f, g, h, a);

    m0 = SHA256_EXPAND(me, m9, m1, m0);
    m1 = SHA256_EXPAND(mf, ma, m2, m1);
    m2 = SHA256_EXPAND(m0, mb, m3, m2);
    m3 = SHA256_EXPAND(m1, mc, m4, m3);
    m4 = SHA256_EXPAND(m2, md, m5, m4);
    m5 = SHA256_EXPAND(m3, me, m6, m5);
    m6 = SHA256_EXPAND(m4, mf, m7, m6);
    m7 = SHA256_EXPAND(m5, m0, m8, m7);
    m8 = SHA256_EXPAND(m6, m1, m9, m8);
    m9 = SHA256_EXPAND(m7, m2, ma, m9);
    ma = SHA256_EXPAND(m8, m3, mb, ma);
    mb = SHA256_EXPAND(m9, m4, mc, mb);
    mc = SHA256_EXPAND(ma, m5, md, mc);
    md = SHA256_EXPAND(mb, m6, me, md);
    me = SHA256_EXPAND(mc, m7, mf, me);
    mf = SHA256_EXPAND(md, m8, m0, mf);

    SHA256_STEP(m0, k[0x20], a, b, c, d, e, f, g, h);
    SHA256_STEP(m1, k[0x21], h, a, b, c, d, e, f, g);
    SHA256_STEP(m2, k[0x22], g, h, a, b, c, d, e, f);
    SHA256_STEP(m3, k[0x23], f, g, h, a, b, c, d, e);
    SHA256_STEP(m4, k[0x24], e, f, g, h, a, b, c, d);
    SHA256_STEP(m5, k[0x25], d, e, f, g, h, a, b, c);
    SHA256_STEP(m6, k[0x26], c, d, e, f, g, h, a, b);
    SHA256_STEP(m7, k[0x27], b, c, d, e, f, g, h, a);
    SHA256_STEP(m8, k[0x28], a, b, c, d, e, f, g, h);
    SHA256_STEP(m9, k[0x29], h, a, b, c, d, e, f, g);
    SHA256_STEP(ma, k[0x2a], g, h, a, b, c, d, e, f);
    SHA256_STEP(mb, k[0x2b], f, g, h, a, b, c, d, e);
    SHA256_STEP(mc, k[0x2c], e, f, g, h, a, b, c, d);
    SHA256_STEP(md, k[0x2d], d, e, f, g, h, a, b, c);
    SHA256_STEP(me, k[0x2e], c, d, e, f, g, h, a, b);
    SHA256_STEP(mf, k[0x2f], b, c, d, e, f, g, h, a);

    m0 = SHA256_EXPAND(me, m9, m1, m0);
    m1 = SHA256_EXPAND(mf, ma, m2, m1);
    m2 = SHA256_EXPAND(m0, mb, m3, m2);
    m3 = SHA256_EXPAND(m1, mc, m4, m3);
    m4 = SHA256_EXPAND(m2, md, m5, m4);
    m5 = SHA256_EXPAND(m3, me, m6, m5);
    m6 = SHA256_EXPAND(m4, mf, m7, m6);
    m7 = SHA256_EXPAND(m5, m0, m8, m7);
    m8 = SHA256_EXPAND(m6, m1, m9, m8);
    m9 = SHA256_EXPAND(m7, m2, ma, m9);
    ma = SHA256_EXPAND(m8, m3, mb, ma);
    mb = SHA256_EXPAND(m9, m4, mc, mb);
    mc = SHA256_EXPAND(ma, m5, md, mc);
    md = SHA256_EXPAND(mb, m6, me, md);
    me = SHA256_EXPAND(mc, m7, mf, me);
    mf = SHA256_EXPAND(md, m8, m0, mf);

    SHA256_STEP(m0, k[0x30], a, b, c, d, e, f, g, h);
    SHA256_STEP(m1, k[0x31], h, a, b, c, d, e, f, g);
    SHA256_STEP(m2, k[0x32], g, h, a, b, c, d, e, f);
    SHA256_STEP(m3, k[0x33], f, g, h, a, b, c, d, e);
    SHA256_STEP(m4, k[0x34], e, f, g, h, a, b, c, d);
    SHA256_STEP(m5, k[0x35], d, e, f, g, h, a, b, c);
    SHA256_STEP(m6, k[0x36], c, d, e, f, g, h, a, b);
    SHA256_STEP(m7, k[0x37], b, c, d, e, f, g, h, a);
    SHA256_STEP(m8, k[0x38], a, b, c, d, e, f, g, h);
    SHA256_STEP(m9, k[0x39], h, a, b, c, d, e, f, g);
    SHA256_STEP(ma, k[0x3a], g, h, a, b, c, d, e, f);
    SHA256_STEP(mb, k[0x3b], f, g, h, a, b, c, d, e);
    SHA256_STEP(mc, k[0x3c], e, f, g, h, a, b, c, d);
    SHA256_STEP(md, k[0x3d], d, e, f, g, h, a, b, c);
    SHA256_STEP(me, k[0x3e], c, d, e, f, g, h, a, b);
    SHA256_STEP(mf, k[0x3f], b, c, d, e, f, g, h, a);

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
