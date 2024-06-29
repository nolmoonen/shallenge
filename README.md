# shallenge

```shell
git clone https://github.com/nolmoonen/shallenge.git
cd shallenge
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

With CUDA 12.4 driver 550.54.14 Ubuntu 22.04.2 x86_64 NVIDIA GeForce RTX 2070:

```shell
$ ./build/shallenge 
3.634897GH/s (16478.255859ms)
nol/000000000000000000000000000000000000AAABAACajbcd
00000019 16613d46 668904df ee5f1393 c8432f2c cc0360f0 b135b43f e8fb50ba
...
final result:
nol/000000000000000000000000000000000000AAAhAGy8jwef
00000000 00041c7a a9916462 0bcc1db7 69d5ca6a a3696963 990e8b51 4cd201db
```
