# Common operator model contract

The library provides 15 integer operators and seven nonlinear references.
These ESL models describe behavior and resource demand. They are not 22 native
opcodes or hardware capabilities of the released W4A4 MVM device.

## Integer operators and layouts

All tensors are contiguous row-major. Images use HWC with batch one.

| Operator | Shape parameters | Layout / output |
| --- | --- | --- |
| dot | K | A[K], B[K] -> [1] |
| gemv | M,K | A[K], B[M,K] -> [M] |
| gemm | M,K,N | A[M,K], B[K,N] -> [M,N] |
| conv1d | L,Cin,Cout,Kw | A[L,Cin], B[Kw,Cin,Cout] -> [L-Kw+1,Cout] |
| conv2d | H,W,Cin,Cout,Kh,Kw | A[H,W,Cin], B[Kh,Kw,Cin,Cout] -> [H-Kh+1,W-Kw+1,Cout] |
| dwconv2d | H,W,C,Kh,Kw | A[H,W,C], B[Kh,Kw,C] -> [H-Kh+1,W-Kw+1,C] |
| fir | L,K | A[L], B[K] -> [L-K+1] |
| add, mul | N | A[N], B[N] -> [N] |
| relu, requant | N | A[N] -> [N] |
| reduce_sum, reduce_max | N | A[N] -> [1] |
| maxpool2d | H,W,C,Kh,Kw,Sh,Sw | VALID pooling, explicit strides |
| transpose | H,W | A[H,W] -> [W,H] |

Convolution means stride-one VALID cross-correlation, without bias, dilation or
group extensions; depthwise multiplier is one. FIR uses
y[t]=sum(B[k]*A[t+K-1-k]); K>L is invalid. Input samples are not counted as a
new external tensor per sliding window.

Operands are signed INT4, except requant input is INT16. MAC/reduce_sum use
ACC16 with bounded legal lengths. MAC/add/mul/reduce_sum output INT16; other
integer operators output INT4. Requant is explicitly
clip(round_away_from_zero(3*x/4),-8,7), not a framework-wide quantization contract.

Count a MAC as one multiply-accumulate. Scalar ALU operations and movement are
separate: sum/max use N-1 operations; pool uses Kh*Kw-1 comparisons/output;
transpose counts bytes but zero MAC/scalar arithmetic. weight_bytes denotes
the second operand uniformly, even when it is not reusable weight data.

## Independent functional and cycle paths

The functional path directly evaluates tensor indexing. The cycle path lowers
outputs into WorkTerms carrying source addresses and actually computes values
at delayed commit. It never copies golden values into a fabricated trace.

Resources include one/two/four lanes, depth two/three/four and explicit A/B read
ports. Each lane has one ACC16; commit latency is depth-1 ticks.
A deterministic lane-priority greedy scheduler counts unique A and B addresses
per tick, broadcasts equal addresses and stalls lanes when port budgets are
exceeded. It is not a global optimal scheduler. Output writes use a modeled
lane-owned bank/channel, whose implementation cost is not free.

Integer end-to-end scenarios serialize input load, core and output readback,
including one START byte and the historical sideband-address assumption.
Resident bytes exclude controller, ports and pipeline area. Large tensors still
need a real tiling/streaming plan.

## Nonlinear reference definitions

Rows of a two-dimensional tensor are processed along the last dimension.
Double mathematical loops are compared against independently executed primitive
DAGs for Softmax, LayerNorm, RMSNorm, GELU, SiLU, Sigmoid and Tanh.

Softmax subtracts row max; LayerNorm uses population variance and affine
gamma/beta; RMSNorm uses mean square and gamma; GELU uses exact erf;
SiLU is x*sigmoid(x); Sigmoid uses a stable exponential reference and an
equivalent tanh DAG. epsilon defaults to 1e-5 and must be finite/positive.
Inputs are finite and bounded by 1e6. Gamma/beta are explicit vectors.

The default comparison tolerance is 1e-9*(1+abs(reference)). LayerNorm also adds
8*DBL_EPSILON*max(abs(x))*max(abs(gamma))/sqrt(epsilon) for conditioning near
large-offset constant rows. This is numerical reference tolerance, not a
hardware approximation-accuracy result.

The DAG has one II=1 ALU (latency one) and one shared nonpipelined SFU.
exp/tanh default to eight cycles; div/sqrt/erf to sixteen.
SFU-scale two doubles these delays for sensitivity. Inputs/constants are ready
at tick zero; first issue is tick one, and issue+latency completion can satisfy
same-tick dependencies. ALU and SFU can overlap.

Temporary-value capacity and data-supply port limits are not modeled. Binary64
reference arithmetic is not proposed FP64 hardware. LUT/iteration precision,
storage, throughput and native nonlinear RTL remain unimplemented.

## Experiment and sources

103 model runs cover 75 integer cases (15 operators x five resources) and
28 nonlinear cases (seven operators x two widths x two SFU scales).
Three templates and scenario points yield 1602 report rows.
Run analyze_operators.py as described in the [modeling guide](modeling.md).

Primary mathematical references:
[Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html),
[LayerNorm](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html),
[RMSNorm](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html),
[GELU](https://onnx.ai/onnx/operators/onnx__Gelu.html).
Only the explicitly described subsets are modeled.
