# Tiny-LPU common operator models

Fifteen integer operators use independent bit-exact references and port-constrained cycle execution. Seven nonlinear operators use binary64 mathematics and an independent primitive DAG. Cycles are ESL assumptions, not physical Fmax.

## Integer operators: two lanes, depth three, one A port and two B ports

Scenarios use 100MHz and a 25MB/s half-duplex host, serialized load/compute/readback, one START byte and the historical sideband-address assumption.

| Operator | Shape | Core clocks | MACs | Scalar ops | Resident-data lower bound B | Core us | Sequential total us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dot | [16] | 18 | 16 | 0 | 18 | 0.180 | 0.940 |
| gemv | [4, 16] | 34 | 64 | 0 | 48 | 0.340 | 2.300 |
| gemm | [4, 8, 4] | 66 | 128 | 0 | 64 | 0.660 | 3.260 |
| conv1d | [10, 2, 4, 3] | 98 | 192 | 0 | 86 | 0.980 | 4.460 |
| conv2d | [5, 5, 2, 4, 3, 3] | 326 | 648 | 0 | 133 | 3.260 | 8.620 |
| dwconv2d | [5, 5, 4, 3, 3] | 326 | 324 | 0 | 140 | 3.260 | 8.900 |
| fir | [16, 8] | 67 | 72 | 0 | 30 | 0.670 | 1.910 |
| add | [16] | 18 | 0 | 16 | 48 | 0.180 | 2.140 |
| mul | [16] | 18 | 0 | 16 | 48 | 0.180 | 2.140 |
| relu | [16] | 18 | 0 | 16 | 16 | 0.180 | 0.860 |
| requant | [16] | 18 | 0 | 16 | 40 | 0.180 | 1.820 |
| reduce_sum | [16] | 18 | 0 | 15 | 10 | 0.180 | 0.620 |
| reduce_max | [16] | 18 | 0 | 15 | 9 | 0.180 | 0.580 |
| maxpool2d | [6, 6, 2, 2, 2, 2, 2] | 74 | 0 | 54 | 45 | 0.740 | 2.580 |
| transpose | [4, 8] | 34 | 0 | 0 | 32 | 0.340 | 1.660 |

Resident bytes exclude control, addressing, pipelines and register ports; they are not tile-fit evidence. weight_bytes names the second operand even for elementwise data.

## Read ports and mapping

Compare one/four A ports with four lanes, depth three and four B ports. Additional ports have implementation cost.

| Operator | A1 clocks | A4 clocks | Core speedup |
| --- | ---: | ---: | ---: |
| gemv | 18 | 18 | 1.00× |
| gemm | 34 | 34 | 1.00× |
| conv2d | 164 | 164 | 1.00× |
| dwconv2d | 326 | 83 | 3.93× |
| fir | 53 | 26 | 2.04× |
| add | 18 | 6 | 3.00× |
| transpose | 34 | 10 | 3.40× |

## Nonlinear 2x8 reference: ALU II=1 and one nonpipelined SFU

ALU latency is one; exp/tanh eight; div/sqrt/erf sixteen clocks. One SFU operation is active at a time; ALU can overlap. Double SFU latency for sensitivity.

| Operator | Default clocks | Double-SFU-latency clocks | Default SFU busy clocks | Primitive counts |
| --- | ---: | ---: | ---: | --- |
| softmax | 178 | 338 | 160 | {"add": 14, "div": 2, "exp": 16, "max": 14, "mul": 16, "sub": 16} |
| layernorm | 147 | 186 | 64 | {"add": 46, "div": 2, "mul": 52, "sqrt": 2, "sub": 16} |
| rmsnorm | 98 | 162 | 64 | {"add": 16, "div": 2, "mul": 50, "sqrt": 2} |
| gelu | 260 | 516 | 256 | {"add": 16, "erf": 16, "mul": 48} |
| silu | 133 | 261 | 128 | {"add": 16, "mul": 48, "tanh": 16} |
| sigmoid | 132 | 260 | 128 | {"add": 16, "mul": 32, "tanh": 16} |
| tanh | 129 | 257 | 128 | {"tanh": 16} |

Double precision is a mathematical reference, not proposed FP64 hardware. Hardware precision, LUT approximations and supply ports are unselected; physical bytes, host latency and INT4 MAC throughput are intentionally unspecified.

## Microarchitecture implications

- GEMV, some GEMM and ordinary convolution can share A per tick and fit output-row parallelism.
- Depthwise convolution, adjacent FIR outputs and elementwise operations often need distinct A addresses. Compare banking, K parallelism or sliding-window support before adding lanes.
- ReLU, comparison, reduction, transpose and requant need explicit ALU, movement or quantization units; do not count them all as MAC/s.
- Softmax and normalization also require reductions, exp/sqrt/division. These models do not imply native nonlinear INT4-PE support.
- All three templates save results; equal ESL resources give equal clocks. Physical cross-process PPA needs separate evidence.

[Full CSV](summary.csv) - [Raw model results](model_results.json) - [Run configuration](run.json)
