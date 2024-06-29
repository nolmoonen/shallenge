# shallenge

```shell
git clone https://github.com/nolmoonen/shallenge.git
cd shallenge
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

- Ubuntu 22.04.2 x86_64, CUDA 12.4 driver 550.54.14, NVIDIA GeForce RTX 2070: ~3.65GH/s
- Ubuntu 22.04.3 x86_64, CUDA 11.8 driver 545.23.08, NVIDIA GeForce RTX 3090: ~7.71GH/s

Best result:

```shell
nol/000000000000000000000000000000000000AAAhAGy8jwef
00000000 00041c7a a9916462 0bcc1db7 69d5ca6a a3696963 990e8b51 4cd201db
```
