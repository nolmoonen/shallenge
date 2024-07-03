# shallenge

CUDA program to find the input that produces the lowest SHA256 hash. See https://shallenge.quirino.net/ ([archive](https://web.archive.org/web/20240703201533/https://shallenge.quirino.net/)) for details.

```shell
git clone https://github.com/nolmoonen/shallenge.git
cd shallenge
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

- Ubuntu 22.04.2 x86_64, CUDA 12.4 driver 550.54.14, NVIDIA GeForce RTX 2070: [3.69, 3.87]GH/s
- Ubuntu 22.04.3 x86_64, CUDA 11.8 driver 545.23.08, NVIDIA GeForce RTX 3090: [7.92, 8.19]GH/s
- Ubuntu 22.04.1 x86_64, CUDA 12.2 driver 535.113.01 NVIDIA GeForce RTX 3050 Ti Laptop: [2.15, 2.25]GH/s

Best result:

```shell
nol/0000000000000000000000000000000000000000AA6VAM2qJgB
00000000 00009989 b927d0c4 e9c15bcc 28b74c2e c58514bc 36d87080 18e652c0
```
