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
        sha256_round(block.arr[i], k[i], a, b, c, d, e, f, g, h);
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
        int i2  = (i + 16 - 2) % 16;
        i2      = i2 < i ? i2 + 16 : i2;
        int i7  = (i + 16 - 7) % 16;
        i7      = i7 < i ? i7 + 16 : i7;
        int i15 = (i + 16 - 15) % 16;
        i15     = i15 < i ? i15 + 16 : i15;
        int i16 = (i + 16 - 16) % 16;
        i16     = i16 < i ? i16 + 16 : i16;
        printf("uint32_t m%d = sha256_update_m_(m%d, m%d, m%d, m%d);\n", 16 + i, i2, i7, i15, i16);
    }
}
