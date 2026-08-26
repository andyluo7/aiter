# aiter mha kernel

this is an example how to benchmark aiter mha fwd/bwd kernel through c++ API: `aiter::mha_fwd`, `aiter::mha_fwd_splitkv`, `aiter::mha_bwd`.

## build and run
We provide a simple script `build_mha.sh` to build the device library as well as a simple executable:
```
# this will build fwd_v3(asm) only
bash build_mha.sh fwd_v3

# this will build bwd_v3(asm) only
bash build_mha.sh bwd_v3

# this will build full fwd(asm + ck)
bash build_mha.sh fwd

# this will build full bwd(asm + ck)
bash build_mha.sh bwd

# this will build full fwd+bwd
bash build_mha.sh
```
Device library `libmha_fwd.so` and `libmha_bwd.so` will be built under current folder, and corresponding executables `benchmark_mha_fwd` and/or `benchmark_mha_bwd` will also be built. You can type `./benchmark_mha_fwd -?` to list all the supported arguments. You can also refer to the `smoke_test_*` script under this folder for a list of quick test.

To benchmark asm kernel, try following commands:
```

# Set this env before you run
export AITER_ASM_DIR={path_to_aiter}/hsa/

# fwd_v3
./benchmark_mha_fwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -lse=1 -fwd_v3=1 -mode=0 -kname=1 -v=0

# bwd_v3 with atomic fp16
./benchmark_mha_bwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=0 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0

# bwd_v3 with atomic fp32
./benchmark_mha_bwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0
```

## how to build/link aiter mha in your c++ project
We recommend you download the source code of `aiter` and put it under the `3rdparty` submodule folder of your project (you don't need to install `aiter`). We use a way simliar to [cpp_extension](https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py) to build the device kernel library without `torch` dependency (you don't need to install `torch`), so it's easy to embed `aiter` into other project.

Basically the build process will be similiar to that inside `build_mha.sh` script.

First, you need to build the device kernel into a `so`, which is done by a python `compile.py` inside this folder.
```
python3 compile.py
```
you can also call this python script from different directory, the generated `.so` will always under current directory.

Second, link the `.so` into your executable and compile. You need specify the correct path through `-L` inorder to link to the device lib. You also need to specify the include directory through `-I`, for this example you need set `$TOP_DIR/csrc/include` for the `aiter` API header, and the dependent ck header `$TOP_DIR/3rdparty/composable_kernel/include` and `$TOP_DIR/3rdparty/composable_kernel/example/ck_tile/01_fmha/`. Please refer to `build_mha.sh` for detailed command


## `aiter::mha_fwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

you can also call the executable `fwd.exe` to check whether the arguments are supported by the asm kernel with the `-is_v3_check=1` condition, try following commands:
```
    ./fwd.exe -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -lse=1 -fwd_v3=1 -mode=0 -kname=1 -v=0 -is_v3_check=1
```
`causal` below always means `window_size_left == -1 && window_size_right == 0`. The asm and opus kernels are compiled for `mask_bottom_right`; `mask_top_left` is only accepted when `seqlen_q == seqlen_k` (the two are equivalent there). `fp8bf16` means fp8 q/k/v with a bf16 output, and it requires the fp32 `q/k/v_descale` buffers to be set.

| data_type    | hdim_q  | hdim_v  | mode           | mask_type                            | general constraints                                | kernel type | mi308 | mi300/325 | mi350/355  |
|--------------|---------|---------|----------------|--------------------------------------|----------------------------------------------------|-------------|-------|-----------|------------|
| bf16         | 128     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | asm         | y     | y         | y          |
| bf16         | 192     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | asm         | y     | y         | y          |
| fp8bf16      | 128     | 128     | batch or group | no_mask or causal(mask_bottom_right) | same as above; descale of q/k/v is required        | asm         | y     | y         | y          |
| fp8bf16      | 256     | 256     | batch or group | no_mask or causal(mask_bottom_right) | same as above; descale of q/k/v is required        | asm         | n     | n         | y          |
| bf16         | 128     | 128     | batch          | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | opus        | n     | n         | y          |
| bf16         | 192     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | opus        | n     | n         | y          |
| fp16 or bf16 | [0,32]  | [0,32]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,64]  | (0,64]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,80]  | (0,96]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,96]  | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,128] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,192] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,192] | (0,192] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,256] | (0,256] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp8bf16      | (0,128] | (0,128] | batch or group | no_mask or causal or swa             | descale of q/k/v is required                       | ck          | y     | y         | y          |
| fp8bf16      | (0,192] | (0,128] | batch or group | no_mask or causal or swa             | descale of q/k/v is required                       | ck          | y     | y         | y          |

Notes:
* The ck rows are matched top-down: the first row whose `hdim_q`/`hdim_v` both fit is the one that gets dispatched.
* `logits_soft_cap` and the attention sink are only implemented by the ck kernels; the asm and opus paths do not guard against them, so pass `fwd_v3=0` (or leave it at the default) when you need them.
* `-v3_bf16_cvt` (0:RTNE, 1:RTNA, 2:RTZ) only affects the gfx942 asm kernels. All three variants exist for `bf16`, while `fp8bf16` on gfx942 only ships the RTNA(=1) variant. gfx950 has a single variant and ignores this flag.
* The opus rows are **not** reachable through `aiter::mha_fwd`. They have their own entry point, `fmha_fwd_bf16_opus_fwd`, which `fwd.exe` calls with `-fwd_v3=2`. bias, dropout, `logits_soft_cap` and the attention sink are not parameters of that entry point at all, so the API cannot be handed them by mistake — but `fwd.exe` still accepts `-bias`, `-p_drop`, `-logits_soft_cap`, `-qscale` and a non-bf16 `-prec` under `-fwd_v3=2` and passes the buffers down unchanged, which makes the reported number describe something other than what was asked for. A head-dim pair outside the two rows above, group mode on the D=128 kernel, and an over-large kv extent are refused and print `not supported yet`.
* The opus kernels are compiled for gfx950 only: on any other arch the kernel template expands to an empty stub, and nothing checks the arch at runtime, so a call there returns without writing `out`.
* The D=128 opus kernel needs the kv byte extent (`seqlen_k * max(k, v seqlen-stride) * 2`) to stay below 2^32, because a larger one wraps the async-load offset. The 192/128 kernel rebases its buffer descriptors per tile and has no such limit.
* q/k/v/out must be contiguous along the head dim; the remaining strides are free, so both bshd and bhsd work. `-vlayout=c` does not (opus reads V row-major over the sequence).


## `aiter::mha_bwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

you can also call the executable `bwd.exe` to check whether the arguments are supported by the asm kernel with the `-v3_api_check=1` condition, try following commands:
```
    ./bwd.exe -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=0 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0 -v3_api_check=1
```
Unlike fwd, the bwd asm kernels have separate `mask_top_left` and `mask_bottom_right` instances, so `causal` below covers both unless stated otherwise. The generic mask (`-mask=g:y,x`) is never supported by asm. `dq_acc` is no longer supplied by the caller: it is allocated internally through `mha_bwd_args::workspace_alloc`.

| data_type    | hdim_q       | hdim_v          | mode           | mask_type                | dq_accumulation          | general constraints                                                       | shape&stride constraints                                                                                                                                                                                          | kernel type(asm/ck) | mi308 | mi300/325 | mi350/355 |
|--------------|--------------|-----------------|----------------|--------------------------|--------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------|-----------|-----------|
| fp16 or bf16 | (128,192]/x8 | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch          | swa                      | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | n         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch          | no_mask or causal_top_left | atomic_f16             | bias, dbias, dropout and deterministic is not supported                   | seqlen_q == seqlen_k and seqlen_k % 64 == 0. The shape&stride of q and do must be the same, the shape&stride of k and v must be the same, and dk/dv must keep the nhead stride of k/v.                             | asm                 | y     | y         | n         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch or group | no_mask or causal        | atomic_f16               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | n     | n         | y         |
| fp16 or bf16 | 192          | 128             | batch          | no_mask or causal        | atomic_f32 or atomic_f16 | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | n     | n         | y         |
| fp16 or bf16 | 64           | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | 64           | equal to hdim_q | batch          | no_mask or causal_top_left | atomic_f16             | bias, dbias, dropout and deterministic is not supported                   | seqlen_q == seqlen_k and seqlen_k % 64 == 0. The shape&stride of q and do must be the same, the shape&stride of k and v must be the same, and dk/dv must keep the nhead stride of k/v.                             | asm                 | y     | y         | y         |
| fp16 or bf16 | [0,32]       | [0,32]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,64]       | (0,64]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,96]       | (0,96]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,128]      | (0,128]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,256]      | (0,256]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |

Notes:
* All asm rows additionally require `hdim_q % 8 == 0 && hdim_v % 8 == 0`. `hdim_q` is padded up to 64/128/192 internally, and the `hdim_q == 64` bucket has no padded-hdim instance, so a head dim below 64 always falls back to ck.
* The rows marked `causal_top_left` have no `mask_bottom_right` instance. On gfx942 a bottom-right causal request is remapped to top-left, which is legal because those rows already require `seqlen_q == seqlen_k`; on gfx950 (`hdim_q == 64`, `atomic_f16`) there is no such remap and the bottom-right case falls back to ck.
* `-v3_bf16_cvt` (0:RTNE, 1:RTNA, 2:RTZ) picks the float→bf16 rounding variant of the bf16 dqdkdv and dq_convert instances. Every gfx942 bf16 instance is rounding-specific; on gfx950 only the `hdim_q == hdim_v == 192` and `hdim_q == hdim_v == 64` dqdkdv instances are, and all the fp16 instances are rounding-agnostic.
* gfx1250 is also dispatched to asm, but only for `bf16`, `hdim_q == hdim_v == 128`, batch mode, `atomic_f32`, `no_mask` or `mask_bottom_right`, and `seqlen_q == seqlen_k` with `seqlen_k % 128 == 0`.


## the asm and opus kernel performance of the attention forwards and attention backwards.
the performance data was tested under the conditions of BF16 and BSHD in batch mode.

The table covers both head-dim pairs the asm forward supports, `hdim_q`/`hdim_v`
of 128/128 and 192/128, and carries three forward numbers per row: the asm kernel
on MI300X and on MI355X, plus the opus kernel (`-fwd_v3=2`) on MI355X. Every cell
is the best of 3 runs, measured with the asm-only builds (`bash build_mha.sh
fwd_v3` / `bash build_mha.sh bwd_v3`):
```
    ./fwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -lse=1 -fwd_v3=1        -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
    ./fwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -lse=1 -fwd_v3=2        -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
    ./bwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -bwd_v3=1 -v3_bf16_cvt=1 -v3_atomic_fp32=0|1 -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
```

`n/a` marks a cell that no kernel can fill rather than one that was skipped. The
opus kernels are compiled for gfx950 only, so they have no MI300X column at all,
and the 192/128 backward asm instances likewise only exist for gfx950, which
leaves the MI300X backward cells empty on those rows.

![causal-fwd-perf picture](images/causal-fwd-perf.png)
![non-causal-fwd-perf picture](images/non-causal-fwd-perf.png)
*Figure 1: Evaluating GQA attention forwards performance at hdim 128/128 under the conditions of batch=8, q_nheads=64 and kv_nheads=8. The third bar is the opus kernel, which exists on MI355X only.*

![causal-bwd-perf picture](images/causal-bwd-perf.png)
![non-causal-bwd-perf picture](images/non-causal-bwd-perf.png)
*Figure 2: Evaluating GQA attention backwards(a16) performance at hdim 128/128 under the conditions of batch=8, q_nheads=64 and kv_nheads=8.*

**More performance test results are shown in the table below:**

| batch | q_nheads | kv_nheads | seqlen_q | seqlen_kv | hdim_q | hdim_v | causal | FWD(TFLOPS) |         |             | BWD-a16(TFLOPS) |          | BWD-a32(TFLOPS) |         |
|-------|----------|-----------|----------|-----------|--------|--------|--------|-------------|---------|-------------|-----------------|----------|-----------------|---------|
|       |          |           |          |           |        |        |        | MI300X      | MI355X  | MI355X-opus | MI300X          | MI355X   | MI300X          | MI355X  |
| 1     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 338.07      | 618.08  | 640.18      | 344.03          | 527.24   | 313.67          | 505.96  |
| 1     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 513.45      | 1131    | 1106.31     | 311.9           | 919.32   | 269.19          | 707.47  |
| 1     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 527.73      | 1165.81 | 1173.22     | 472.01          | 1066.46  | 423.53          | 789.01  |
| 1     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 558.17      | 1345.15 | 1315.78     | 524.15          | 1195.55  | 481.28          | 822.25  |
| 1     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 549.73      | 1368.78 | 1321.65     | 536.48          | 1192.5   | 491.28          | 830.41  |
| 4     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 458.41      | 1011.71 | 959.23      | 390.4           | 832.21   | 353.44          | 669.55  |
| 4     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 504.8       | 1052.48 | 1045.51     | 459.52          | 985.15   | 430.81          | 745.72  |
| 4     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 577.16      | 1303.75 | 1263.94     | 505.82          | 1143.05  | 457.38          | 804.06  |
| 4     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 574.62      | 1391.74 | 1303.47     | 491.07          | 1207.72  | 458.72          | 831.5   |
| 4     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 584.66      | 1365.63 | 1320.21     | 535.92          | 1216.57  | 476.64          | 840.8   |
| 8     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 459.43      | 1000.55 | 974.3       | 379.88          | 817.38   | 329.69          | 665.39  |
| 8     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 543.77      | 1130.99 | 1089.02     | 475.12          | 1040.38  | 426.56          | 758.79  |
| 8     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 567.82      | 1339.02 | 1284.67     | 519.34          | 1157.42  | 460.44          | 813.3   |
| 8     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 585.29      | 1358.5  | 1318.39     | 518.07          | 1207.14  | 475.56          | 835.84  |
| 8     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 577.5       | 1369.63 | 1328.15     | 534.98          | 1215.84  | 480.87          | 842.14  |
| 1     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 418.36      | 979.25  | 972.92      | 292.68          | 877.53   | 266.06          | 656.64  |
| 1     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 485.45      | 1153.95 | 1110.92     | 437.26          | 915.31   | 393.6           | 720.28  |
| 1     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 546.34      | 1217.35 | 1197.71     | 524.33          | 1121.22  | 470.15          | 794.69  |
| 1     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 591.37      | 1367.23 | 1311.49     | 473             | 1209.98  | 441.82          | 826.49  |
| 1     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 572.09      | 1395.11 | 1312.03     | 503.78          | 1190.89  | 460             | 834.74  |
| 4     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 440.07      | 1017.65 | 970.5       | 376.75          | 812.18   | 340.25          | 662.11  |
| 4     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 554.8       | 1141.57 | 1110.47     | 477.46          | 1039.95  | 425.74          | 757.3   |
| 4     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 573.6       | 1334.37 | 1285.48     | 510.76          | 1161.21  | 456.78          | 812.35  |
| 4     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 592.16      | 1354.33 | 1319.83     | 511.65          | 1207.3   | 468.71          | 836.09  |
| 4     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 578.93      | 1363.39 | 1329.36     | 535.75          | 1215.79  | 479.52          | 840.18  |
| 8     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 466.21      | 930.43  | 884.53      | 389.97          | 896.87   | 357.82          | 674.9   |
| 8     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 556.35      | 1222.56 | 1200.2      | 479.74          | 1060.47  | 430.07          | 768.33  |
| 8     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 578.99      | 1341.97 | 1281.57     | 482.86          | 1133.4   | 445.73          | 816.24  |
| 8     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 577.45      | 1364.85 | 1329.1      | 537.04          | 1205.71  | 475.07          | 836.31  |
| 8     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 571.39      | 1383.15 | 1343.94     | 550.19          | 1210.75  | 480.35          | 845.6   |
| 1     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 383.85      | 989.17  | 973.47      | 291.27          | 882.18   | 264.63          | 651.76  |
| 1     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 506.89      | 1138.65 | 1119.06     | 443.31          | 919.74   | 396.33          | 729.71  |
| 1     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 549.2       | 1227.81 | 1200.99     | 520.99          | 1127.74  | 467.24          | 794.29  |
| 1     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 591.77      | 1376.01 | 1311.64     | 465.87          | 1208.58  | 439.94          | 826.85  |
| 1     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 571.59      | 1391.96 | 1312.4      | 505.49          | 1220.42  | 459.64          | 836.4   |
| 4     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 460.34      | 1011.41 | 963.29      | 395.21          | 820.62   | 332.54          | 663.53  |
| 4     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 556.35      | 1158.81 | 1129.84     | 474.83          | 1033.21  | 424.12          | 756.71  |
| 4     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 575.69      | 1334.24 | 1290.66     | 519.08          | 1158.91  | 457.51          | 811.16  |
| 4     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 590.93      | 1352.78 | 1318.35     | 513.66          | 1206.87  | 469.72          | 836.53  |
| 4     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 582.64      | 1361.86 | 1329.28     | 534.39          | 1214.5   | 475.49          | 841.71  |
| 8     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 497.15      | 968.58  | 917.22      | 389.54          | 885.91   | 360.39          | 683.27  |
| 8     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 556.22      | 1221.03 | 1200.72     | 478.01          | 1062.85  | 426.77          | 768.9   |
| 8     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 581.34      | 1353.66 | 1281        | 481.35          | 1161.45  | 438.77          | 815.89  |
| 8     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 583.23      | 1363.81 | 1330.55     | 536.72          | 1206.4   | 475.68          | 836.55  |
| 8     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 566.17      | 1378.5  | 1346.54     | 550.05          | 1208.77  | 478.88          | 845.86  |
| 1     | 64       | 8         | 16384    | 16384     | 128    | 128    | 0      | 547.78      | 1358.82 | 1334.28     | 519.21          | 1233.04  | 441.55          | 844.28  |
| 1     | 64       | 4         | 16384    | 16384     | 128    | 128    | 0      | 549.09      | 1357.12 | 1335.71     | 516.26          | 1211.63  | 448.83          | 844.39  |
| 1     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 130.62      | 234.47  | 331.76      | 177.565         | 216.09   | 166.78          | 208.185 |
| 1     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 255.105     | 578.1   | 704.84      | 317.3           | 513.72   | 295.865         | 471.98  |
| 1     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 467.805     | 1104.96 | 924.71      | 317.685         | 869.19   | 296.025         | 714.5   |
| 1     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 522.68      | 1180.47 | 1106.31     | 436.13          | 1072.28  | 388.235         | 777.76  |
| 1     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 440.12      | 1177.77 | 1174.4      | 513.85          | 1037.73  | 244.705         | 765.455 |
| 4     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 334.005     | 707.4   | 598.47      | 257.115         | 587      | 226.39          | 489.62  |
| 4     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 419.435     | 850.43  | 807.97      | 377.51          | 751.69   | 330.23          | 599.415 |
| 4     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 486.73      | 1071.33 | 1023.07     | 464.83          | 970.28   | 416.54          | 727.925 |
| 4     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 547.09      | 1302.93 | 1216.16     | 468.205         | 1101.185 | 422.835         | 780.87  |
| 4     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 527.705     | 1319.19 | 1233.87     | 474.205         | 1127.54  | 432.545         | 804.78  |
| 8     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 311.385     | 718.67  | 682.98      | 301.495         | 542.505  | 258.26          | 461.085 |
| 8     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 412.99      | 829.17  | 778.54      | 374.255         | 806.615  | 326.355         | 624.69  |
| 8     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 513.1       | 1141.34 | 1132.25     | 454.36          | 996.29   | 409.05          | 734.075 |
| 8     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 537.36      | 1305.2  | 1231.47     | 491.78          | 1104.53  | 441.4           | 785.89  |
| 8     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 556.045     | 1321.17 | 1257.88     | 495.15          | 1128.37  | 443.78          | 802.775 |
| 1     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 228.54      | 426.39  | 573.65      | 283.58          | 390.355  | 242.43          | 374.255 |
| 1     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 392.425     | 940.54  | 836.65      | 279.72          | 707.55   | 257.855         | 607.725 |
| 1     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 474.385     | 986.72  | 934.23      | 420.265         | 946.565  | 378.155         | 713.03  |
| 1     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 518.29      | 1261.28 | 1207.9      | 481.895         | 1091.225 | 433.285         | 773.91  |
| 1     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 510.895     | 1268.63 | 1230.35     | 501.055         | 1123.16  | 447.995         | 792.56  |
| 4     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 326.51      | 705.51  | 652.43      | 311.005         | 546      | 266.9           | 463.625 |
| 4     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 425.735     | 819.54  | 800.35      | 377.225         | 778.245  | 326.805         | 626.28  |
| 4     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 513.79      | 1158.88 | 1141.32     | 449             | 998.1    | 391.235         | 731.07  |
| 4     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 540.515     | 1306.37 | 1233.32     | 482.505         | 1104.505 | 434.645         | 782.055 |
| 4     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 557.475     | 1321.14 | 1260.58     | 493.745         | 1128.05  | 442.51          | 797.255 |
| 8     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 321.865     | 637.33  | 566.48      | 324.22          | 591.815  | 265.08          | 482.29  |
| 8     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 452.03      | 911.56  | 909.17      | 382.1           | 840.03   | 347.89          | 640.345 |
| 8     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 509.255     | 1187.64 | 1182.48     | 457.05          | 1009.94  | 402.18          | 733.31  |
| 8     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 550.67      | 1304.18 | 1254.67     | 474.02          | 1103.63  | 432.715         | 784.87  |
| 8     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 547.05      | 1332.9  | 1276.6      | 489.075         | 1125.09  | 439.785         | 806.685 |
| 1     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 229.09      | 425.12  | 572.43      | 265.11          | 393.255  | 238.755         | 378.355 |
| 1     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 407.525     | 946.76  | 836.24      | 277.86          | 724.535  | 254.375         | 607.995 |
| 1     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 476.26      | 990.59  | 947.2       | 418.73          | 940.835  | 384.585         | 709.135 |
| 1     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 519.32      | 1271.33 | 1206.26     | 480.06          | 1091.47  | 442.955         | 773.415 |
| 1     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 515.275     | 1319.44 | 1228.87     | 499.72          | 1122.01  | 459.745         | 794.215 |
| 4     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 314.82      | 728.86  | 672.43      | 324.22          | 556.94   | 264.795         | 467.965 |
| 4     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 426.77      | 856.4   | 822.93      | 374.96          | 807.005  | 331.95          | 623.22  |
| 4     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 524.585     | 1166.02 | 1145.03     | 453.97          | 996.42   | 405.02          | 728.495 |
| 4     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 540.935     | 1302.82 | 1233.15     | 478.735         | 1100.705 | 430.95          | 780.27  |
| 4     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 560.63      | 1341.75 | 1259.83     | 491.435         | 1127.945 | 441.345         | 800.555 |
| 8     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 348.76      | 624.18  | 604.07      | 315.035         | 587.465  | 267.48          | 488.92  |
| 8     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 461.89      | 935.33  | 911.07      | 400.31          | 843.345  | 352.7           | 640.085 |
| 8     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 513.795     | 1195.46 | 1180.62     | 456.415         | 1010.345 | 402.68          | 732.155 |
| 8     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 552.78      | 1314.78 | 1251.47     | 473.41          | 1104.48  | 434.51          | 783.295 |
| 8     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 548.65      | 1340.64 | 1275.47     | 488.145         | 1124.74  | 435.745         | 804.075 |
| 1     | 64       | 8         | 16384    | 16384     | 128    | 128    | 1      | 541.55      | 1369.28 | 1259.34     | 458.075         | 1158.875 | 412.04          | 814.065 |
| 1     | 64       | 4         | 16384    | 16384     | 128    | 128    | 1      | 544.1       | 1367.99 | 1260.28     | 458.065         | 1158.905 | 419.975         | 816.12  |
| 1     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 375.85      | 878.1   | 677.09      | n/a             | 550.53   | n/a             | 393.93  |
| 1     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 482.74      | 1057.61 | 1124.45     | n/a             | 644.36   | n/a             | 465.56  |
| 1     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 494.19      | 1039.99 | 1144.88     | n/a             | 887.21   | n/a             | 513.49  |
| 1     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 575.51      | 1236.09 | 1345.1      | n/a             | 952.7    | n/a             | 532.74  |
| 1     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 557.43      | 1227.19 | 1325.81     | n/a             | 1006.75  | n/a             | 545.79  |
| 4     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 437.19      | 947.53  | 992.46      | n/a             | 608.85   | n/a             | 433.72  |
| 4     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 508.79      | 966.93  | 1069.68     | n/a             | 834.43   | n/a             | 499.66  |
| 4     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 559.41      | 1210.58 | 1304.66     | n/a             | 969.08   | n/a             | 525.25  |
| 4     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 565.42      | 1218.83 | 1314.65     | n/a             | 1031.79  | n/a             | 547.23  |
| 4     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 558.48      | 1228.53 | 1330.6      | n/a             | 1051.31  | n/a             | 547.94  |
| 8     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 431.01      | 870.38  | 943.63      | n/a             | 678.85   | n/a             | 439.13  |
| 8     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 533.17      | 1048.03 | 1123.84     | n/a             | 911.26   | n/a             | 508.36  |
| 8     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 565.67      | 1206.65 | 1311.47     | n/a             | 956.37   | n/a             | 528.38  |
| 8     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 558.15      | 1227.05 | 1328.1      | n/a             | 1049.7   | n/a             | 548.75  |
| 8     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 571.3       | 1239.71 | 1341.8      | n/a             | 1050.12  | n/a             | 549.79  |
| 1     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 401.92      | 926.44  | 984.54      | n/a             | 545.33   | n/a             | 411.38  |
| 1     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 434.98      | 1017.25 | 1132.64     | n/a             | 773.28   | n/a             | 497.16  |
| 1     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 547.17      | 1150.91 | 1244.51     | n/a             | 927.32   | n/a             | 511.71  |
| 1     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 557         | 1214.74 | 1315.58     | n/a             | 1001.68  | n/a             | 545.5   |
| 1     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 566.67      | 1219.84 | 1318.51     | n/a             | 1016.15  | n/a             | 544.6   |
| 4     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 436.5       | 903.02  | 957.67      | n/a             | 687.1    | n/a             | 439.57  |
| 4     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 523.61      | 1060.55 | 1160.87     | n/a             | 900.64   | n/a             | 510.24  |
| 4     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 555.97      | 1201.73 | 1292.21     | n/a             | 963.01   | n/a             | 528.62  |
| 4     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 548.34      | 1226.97 | 1328.93     | n/a             | 1049.4   | n/a             | 549.57  |
| 4     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 536.27      | 1237.55 | 1342.34     | n/a             | 1050.08  | n/a             | 549.91  |
| 8     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 460.59      | 883.54  | 940.55      | n/a             | 728.74   | n/a             | 442.19  |
| 8     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 530.1       | 1155.49 | 1232.92     | n/a             | 917.23   | n/a             | 513.13  |
| 8     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 550.98      | 1203.48 | 1293.68     | n/a             | 986.64   | n/a             | 531.11  |
| 8     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 540.17      | 1236.98 | 1338.86     | n/a             | 1048.26  | n/a             | 550.97  |
| 8     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 530.87      | 1266.42 | 1369.47     | n/a             | 1050.46  | n/a             | 551.45  |
| 1     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 418.14      | 931.26  | 987.35      | n/a             | 551.57   | n/a             | 412.38  |
| 1     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 455.99      | 1024.34 | 1134.04     | n/a             | 789.03   | n/a             | 500.77  |
| 1     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 548.24      | 1165.18 | 1248.2      | n/a             | 927.41   | n/a             | 510.45  |
| 1     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 559.25      | 1211.37 | 1317.49     | n/a             | 1007.26  | n/a             | 546.46  |
| 1     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 574.18      | 1220.28 | 1320.45     | n/a             | 1009.8   | n/a             | 544.75  |
| 4     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 448.71      | 867.96  | 998.78      | n/a             | 677.57   | n/a             | 439.58  |
| 4     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 531.22      | 1071.24 | 1160.51     | n/a             | 912.26   | n/a             | 508.39  |
| 4     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 565.29      | 1201.55 | 1294.64     | n/a             | 962.01   | n/a             | 528.86  |
| 4     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 559.48      | 1227.23 | 1330.05     | n/a             | 1049.2   | n/a             | 549.04  |
| 4     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 572.1       | 1237.49 | 1341.97     | n/a             | 1051.96  | n/a             | 546.59  |
| 8     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 480.31      | 874.16  | 929.28      | n/a             | 732.01   | n/a             | 443.04  |
| 8     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 544.32      | 1136.12 | 1259.77     | n/a             | 916.14   | n/a             | 512.91  |
| 8     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 555.52      | 1203.33 | 1294.16     | n/a             | 982.52   | n/a             | 531.09  |
| 8     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 572.77      | 1237.13 | 1339.53     | n/a             | 1047.73  | n/a             | 550.6   |
| 8     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 578.38      | 1266.56 | 1371.44     | n/a             | 1049.71  | n/a             | 551.66  |
| 1     | 64       | 8         | 16384    | 16384     | 192    | 128    | 0      | 552.78      | 1240.94 | 1345.92     | n/a             | 1045.98  | n/a             | 549.78  |
| 1     | 64       | 4         | 16384    | 16384     | 192    | 128    | 0      | 570.23      | 1240.93 | 1346.21     | n/a             | 1045.42  | n/a             | 549.91  |
| 1     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 251.14      | 348.8   | 343.22      | n/a             | 290.9    | n/a             | 273.52  |
| 1     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 391.96      | 778.8   | 580.52      | n/a             | 515.2    | n/a             | 431.43  |
| 1     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 453.39      | 774.56  | 1066.94     | n/a             | 678.79   | n/a             | 472.33  |
| 1     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 541.39      | 964.25  | 1194.66     | n/a             | 904.5    | n/a             | 514.59  |
| 1     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 560.41      | 1011.61 | 1176.32     | n/a             | 880.71   | n/a             | 514.32  |
| 4     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 354.29      | 630.98  | 714.68      | n/a             | 444.13   | n/a             | 346.93  |
| 4     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 406.51      | 670.29  | 808.69      | n/a             | 668.39   | n/a             | 444.2   |
| 4     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 522.02      | 895.57  | 1082.16     | n/a             | 846.45   | n/a             | 489.41  |
| 4     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 550.77      | 1043.92 | 1277.76     | n/a             | 969.22   | n/a             | 531     |
| 4     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 551.88      | 1059.32 | 1271.44     | n/a             | 971.64   | n/a             | 536.47  |
| 8     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 348.14      | 589.9   | 692.73      | n/a             | 505.67   | n/a             | 370.35  |
| 8     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 430.64      | 732.11  | 839.61      | n/a             | 714.8    | n/a             | 448.05  |
| 8     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 523.1       | 947.21  | 1169.76     | n/a             | 893.27   | n/a             | 503.59  |
| 8     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 541.58      | 1051.69 | 1259.07     | n/a             | 970.86   | n/a             | 533.37  |
| 8     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 547.89      | 1062.98 | 1281.95     | n/a             | 998.93   | n/a             | 539.72  |
| 1     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 319         | 611.54  | 516.91      | n/a             | 478.61   | n/a             | 354.36  |
| 1     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 440.65      | 761.97  | 944.11      | n/a             | 563      | n/a             | 421.84  |
| 1     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 486.59      | 840.19  | 1008.04     | n/a             | 816.23   | n/a             | 492.31  |
| 1     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 553.13      | 1012.09 | 1278.06     | n/a             | 903.11   | n/a             | 514.05  |
| 1     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 559.31      | 1044.45 | 1285.9      | n/a             | 954.18   | n/a             | 535.18  |
| 4     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 358.89      | 585.4   | 712.78      | n/a             | 510.98   | n/a             | 359.55  |
| 4     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 431.39      | 737.39  | 863.61      | n/a             | 714.16   | n/a             | 448.51  |
| 4     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 518.09      | 947.17  | 1185.61     | n/a             | 892.4    | n/a             | 502.65  |
| 4     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 539.21      | 1052.5  | 1255.41     | n/a             | 970.8    | n/a             | 533.46  |
| 4     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 540.14      | 1066.27 | 1280.38     | n/a             | 999.1    | n/a             | 538.97  |
| 8     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 350.34      | 538.42  | 625.19      | n/a             | 553.09   | n/a             | 374.71  |
| 8     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 469.64      | 788.42  | 934.26      | n/a             | 749.18   | n/a             | 458.29  |
| 8     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 515.66      | 997.07  | 1214.48     | n/a             | 906.21   | n/a             | 506.17  |
| 8     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 530.47      | 1061.13 | 1269.98     | n/a             | 979.59   | n/a             | 535.99  |
| 8     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 541.1       | 1022.7  | 1294.81     | n/a             | 999.4    | n/a             | 541.99  |
| 1     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 269.09      | 615.6   | 516.02      | n/a             | 492.59   | n/a             | 360.21  |
| 1     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 456.89      | 776.27  | 953.08      | n/a             | 560.56   | n/a             | 420.24  |
| 1     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 504.33      | 843.26  | 1030.25     | n/a             | 812.04   | n/a             | 491.91  |
| 1     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 561.86      | 1010.49 | 1282.23     | n/a             | 902.34   | n/a             | 513.04  |
| 1     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 566.33      | 1045.14 | 1286.96     | n/a             | 952.2    | n/a             | 536.44  |
| 4     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 367.64      | 606.73  | 715.23      | n/a             | 518.17   | n/a             | 366.31  |
| 4     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 444.97      | 743.89  | 869.35      | n/a             | 716.26   | n/a             | 447.66  |
| 4     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 531.91      | 953.04  | 1186.81     | n/a             | 905.57   | n/a             | 501.92  |
| 4     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 550.59      | 1051.86 | 1260.47     | n/a             | 970.46   | n/a             | 532.92  |
| 4     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 553.92      | 1065.26 | 1283.39     | n/a             | 1000.56  | n/a             | 539.56  |
| 8     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 359.01      | 542.67  | 625.91      | n/a             | 559.32   | n/a             | 373.54  |
| 8     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 479.86      | 802.3   | 958.16      | n/a             | 749.95   | n/a             | 456.45  |
| 8     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 529.82      | 1001.83 | 1205.65     | n/a             | 895.9    | n/a             | 507.02  |
| 8     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 541.56      | 1064.67 | 1269.44     | n/a             | 981.01   | n/a             | 535.99  |
| 8     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 550.55      | 1029.77 | 1293.82     | n/a             | 998.76   | n/a             | 542.05  |
| 1     | 64       | 8         | 16384    | 16384     | 192    | 128    | 1      | 560.35      | 1073.37 | 1299.01     | n/a             | 1006.64  | n/a             | 545.9   |
| 1     | 64       | 4         | 16384    | 16384     | 192    | 128    | 1      | 566.76      | 1073.77 | 1297.1      | n/a             | 1004.62  | n/a             | 546.33  |

