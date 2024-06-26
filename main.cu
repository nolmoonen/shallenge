#include <cstdint>
#include <cstring>
#include <iostream>

constexpr uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr int block_size_u32 = 512 / 32;
constexpr int hash_size_u32  = 256 / 32;

uint32_t rotr(uint32_t a, int b) { return (a >> b) | (a << (32 - b)); }
uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

uint32_t swap_endian(uint32_t x)
{
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&x);
    return uint32_t{ptr[3]} | (uint32_t{ptr[2]} << 8) | (uint32_t{ptr[1]} << 16) |
           (uint32_t{ptr[0]} << 24);
}

void sha256(uint32_t (&hash)[hash_size_u32], const uint32_t (&block)[block_size_u32])
{
    uint32_t m[64];
    for (int i = 0; i < block_size_u32; ++i) {
        m[i] = swap_endian(block[i]);
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
    }

    hash[0] = 0x6a09e667;
    hash[1] = 0xbb67ae85;
    hash[2] = 0x3c6ef372;
    hash[3] = 0xa54ff53a;
    hash[4] = 0x510e527f;
    hash[5] = 0x9b05688c;
    hash[6] = 0x1f83d9ab;
    hash[7] = 0x5be0cd19;

    uint32_t a = hash[0];
    uint32_t b = hash[1];
    uint32_t c = hash[2];
    uint32_t d = hash[3];
    uint32_t e = hash[4];
    uint32_t f = hash[5];
    uint32_t g = hash[6];
    uint32_t h = hash[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[i] + m[i];
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

    hash[0] = swap_endian(hash[0] + a);
    hash[1] = swap_endian(hash[1] + b);
    hash[2] = swap_endian(hash[2] + c);
    hash[3] = swap_endian(hash[3] + d);
    hash[4] = swap_endian(hash[4] + e);
    hash[5] = swap_endian(hash[5] + f);
    hash[6] = swap_endian(hash[6] + g);
    hash[7] = swap_endian(hash[7] + h);
}

void do_hash(const uint32_t (&block)[block_size_u32], uint32_t (&hash)[hash_size_u32])
{
    sha256(hash, block);
}

int main()
{
    const char* message_string = "test/0";
    const int message_len      = std::strlen(message_string);
    if (message_len + 1 + 8 > 512 / 8) {
        std::cout << "message too long\n";
        return EXIT_FAILURE;
    }

    uint32_t block[block_size_u32] = {}; // zero-initialize
    for (int i = 0; i < message_len; ++i) {
        reinterpret_cast<uint8_t*>(block)[i] = message_string[i];
    }
    reinterpret_cast<uint8_t*>(block)[message_len] = 0x80; // append single 1 bit
    // final 64-bits is length of original message
    uint32_t lenu32     = 8 * message_len;
    auto tmp            = reinterpret_cast<const uint8_t*>(&lenu32);
    block[512 / 32 - 1] = tmp[3] | (tmp[2] << 8) | (tmp[1] << 16) | (tmp[0] << 24);

    uint32_t hash[hash_size_u32];
    sha256(hash, block);

    // 582fa6476ded1ee8e6f6f1339d98e5a1b609666b5ab07521bfb3a25b1502ff63
    for (int i = 0; i < 256 / 8; ++i) {
        printf("%x", reinterpret_cast<const uint8_t*>(hash)[i]);
    }
    printf("\n");
}
