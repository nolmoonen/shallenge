#include "common.cuh"

#include <stdio.h>

int main()
{
    block_t block;
    set_common(block);

    uint32_t a = aa;
    uint32_t b = bb;
    uint32_t c = cc;
    uint32_t d = dd;
    uint32_t e = ee;
    uint32_t f = ff;
    uint32_t g = gg;
    uint32_t h = hh;

    for (int i = 0; i < 10; ++i) {
        sha256_round(block.arr, a, b, c, d, e, f, g, h, 0, i);
    }

    printf("uint32_t a = 0x%08x;\n", a);
    printf("uint32_t b = 0x%08x;\n", b);
    printf("uint32_t c = 0x%08x;\n", c);
    printf("uint32_t d = 0x%08x;\n", d);
    printf("uint32_t e = 0x%08x;\n", e);
    printf("uint32_t f = 0x%08x;\n", f);
    printf("uint32_t g = 0x%08x;\n", g);
    printf("uint32_t h = 0x%08x;\n", h);

    for (int i = 0; i < 16; ++i) {
        const int i2  = (i + 16 - 2) % 16;
        const int i7  = (i + 16 - 7) % 16;
        const int i15 = (i + 16 - 15) % 16;
        const int i16 = (i + 16 - 16) % 16;
        printf(
            "constexpr uint32_t m%d = 0x%08x; // f(m%d, m%d, m%d, m%d)\n",
            16 + i,
            block.arr[i] =
                sha256_update_m_(block.arr[i2], block.arr[i7], block.arr[i15], block.arr[i16]),
            i2,
            i7,
            i15,
            i16);
    }
}
