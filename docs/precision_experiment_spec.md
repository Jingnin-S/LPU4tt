# Integer and mixed-precision reference experiment contract

This study preserves historical INT4/ACC16 ABI-v1 and compares independent
numeric modes. The later base ISA/device is INT4/ACC32. Reference support must
not be mistaken for native device capability.

## Types and arithmetic

NumericMode declares activation, weight, accumulator and output dtype separately.
Modes include INT4/ACC16/OUT16, INT4/ACC32/OUT32, INT8/ACC32/OUT32,
FP16/FP32 with OUT16 or OUT32, BF16/FP32 with OUT16 or OUT32, and FP32/FP32.
Input/output width fields in the older IR are not reinterpreted.

FP16/BF16 inputs use RNE conversion from FP32, retaining subnormals. Each product
is rounded to FP32 and followed by a separate FP32 addition in increasing K
order. Fast-math and FMA contraction are disabled. Output rounding is separate.
Class, infinity sign, finite encodings and signed zero are checked against
independent PyTorch conversion; NaN payload is unspecified.

Integer references use specified 16/32-bit two's-complement wrap and count
pre-wrap range violations. This diagnoses unsafe accumulation; it is distinct
from the later frozen ISA-v0.1 wrap32 semantics.

## Quantization

Signed raw ranges are [-8,7] and [-128,127], with zero-point zero.
FP32 scale = absmax/qmax; zero tensors use scale=1.
Quantization is clip(RNE(x/scale),qmin,qmax). Negative absmax therefore normally
maps to -qmax; raw integer diagnostics separately exercise -8 and -128.

W is offline per-output-channel. A is either dynamic per frame or static from
32 independent FP32 calibration frames. Static calibration never reads test
inputs. outlier_shift deliberately includes unseen test outliers.
Quantization/dequantization and scales belong to the host; their runtime cost
is not device cycles. Existing HOST_RELU_Q keeps its original half-up contract.

## Workloads and checks

Seven graphs: Linear16->8, 16->7, 17->3, 64->8, 256->8, 1024->4,
and Linear16->8->ReLU->4. Seven seeded distributions: normal, small (~1e-9),
large (~1e5), wide_dynamic, outlier_shift, cancellation and zero.
Each fixture has 16 test inputs and 32 independent calibration inputs.
All modes share original FP32 tensors and source hashes.

Six integer/scale combinations and five floating modes give 539 rows across
49 fixtures. Framework FP32 is the comparison baseline; FP64 is diagnostic.
C++ results/counters are checked against independent integer oracles or
Torch conversion plus sequential NumPy FP32 arithmetic. Framework GEMM
reduction/FMA order may differ and is measured rather than assumed equivalent.

Metrics include absolute error, RMSE, NRMSE and cosine, input Inf/zero conversion,
nonfinite product/accumulation/output and integer overflow events. A nonfinite
result makes the global error null; finite subsets are separately reported.
Zero reference and zero error gives NRMSE=0; zero reference and nonzero error
gives null. Conversion counts and per-MAC event counts are distinct.

There are 84 actual historical ABI-v1 layer cross-checks for supported
INT4/ACC16 K<=64 layers. Longer K and other formats use the independent numeric
reference. Raw extrema exercise INT4 K64/256/512 and INT8 K2/64/4096.
Conversion tests cover all 65536 FP16/BF16 encodings, 50000 random FP32 bit
patterns, ties, subnormals, signed zero, infinity and NaN.

Storage budgets include row packing, lane padding, output alignment and slot
count, while host scales and accumulator registers are listed separately.
They are not synthesis area. This synthetic study makes no pretrained-model,
perplexity, task-accuracy or format-specific Fmax/power claim.
See [results](precision_results.md) and [reproduction](modeling.md).
