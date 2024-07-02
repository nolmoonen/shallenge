#ifndef SHALLENGE_COMMON_CUH_
#define SHALLENGE_COMMON_CUH_

#include <cuda_runtime.h>

#include <stdint.h>

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

constexpr int block_size_u32 = 512 / 32;
constexpr int hash_size_u32  = 256 / 32;

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

__forceinline__ __device__ __host__ void set_common(block_t& block)
{
    block.arr[0]  = m00;
    block.arr[1]  = m01;
    block.arr[2]  = m02;
    block.arr[3]  = m03;
    block.arr[4]  = m04;
    block.arr[5]  = m05;
    block.arr[6]  = m06;
    block.arr[7]  = m07;
    block.arr[8]  = m08;
    block.arr[9]  = m09;
    block.arr[10] = m10;
    // block.arr[11] iteration index
    // block.arr[12] thread index
    // block.arr[13] determined by thread
    block.arr[14] = m14;
    block.arr[15] = m15;
}

#endif // SHALLENGE_COMMON_CUH_
