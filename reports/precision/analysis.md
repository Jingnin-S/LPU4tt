# Integer and mixed-precision numerical experiment

INT4/INT8/FP16/BF16 share FP32 source inputs and weights; PyTorch FP32 is the baseline, with FP64 diagnostics.
C++ values/counters are cross-checked against independent Python integer arithmetic or Torch conversion plus sequential NumPy FP32.
FP16/BF16 use RNE inputs, retain subnormals and use separate sequential FP32 multiply/add; OUT16 and OUT32 are separate.

49 workload/distribution fixtures, 539 mode/quantization rows, 16 inputs per fixture, 84 historical ABI-v1 layer validations.

| Mode | Activation quantization | Normal median NRMSE | Small median NRMSE | Nonfinite-result rows |
| --- | --- | ---: | ---: | ---: |
| int4_acc16 | dynamic_absmax | 0.142152 | 0.164274 | 0 |
| int4_acc16 | calibrated_static | 0.166195 | 0.191644 | 0 |
| int4_acc32 | dynamic_absmax | 0.142152 | 0.164274 | 0 |
| int4_acc32 | calibrated_static | 0.166195 | 0.191644 | 0 |
| int8_acc32 | dynamic_absmax | 0.00871279 | 0.00849019 | 0 |
| int8_acc32 | calibrated_static | 0.0128753 | 0.0119541 | 0 |
| fp16_acc32_out32 | none | 0.000304604 | 1 | 8 |
| bf16_acc32_out32 | none | 0.00230954 | 0.00233078 | 0 |
| fp16_acc32_out16 | none | 0.000349317 | 1 | 14 |
| bf16_acc32_out16 | none | 0.00280517 | 0.0028765 | 0 |
| fp32_ordered | none | 9.55918e-08 | 7.86645e-08 | 0 |

NRMSE=norm(candidate-reference)/norm(reference). Global error is null for NaN/Inf results; finite subsets are reported separately rather than discarding failures.

Dynamic A scale requires a host scan; static A scale uses independent FP32 calibration activations. W scale is offline per-output-channel, zero-point zero, RNE and FP32 scale storage. outlier_shift includes unseen test outliers.
All distributions are synthetic and do not establish LLM perplexity, task accuracy or real-model calibration. The MLP is ordinary Linear-ReLU-Linear, distinct from the fixed-3/4 toy graph.

## Integer accumulation range

| Operand type | K | Accumulator bits | Mathematical result | Model result | Overflow events |
| --- | ---: | ---: | ---: | ---: | ---: |
| int4 | 64 | 16 | 4096 | 4096 | 0 |
| int4 | 256 | 16 | 16384 | 16384 | 0 |
| int4 | 512 | 16 | 32768 | -32768 | 1 |
| int4 | 512 | 32 | 32768 | 32768 | 0 |
| int8 | 2 | 16 | 32768 | -32768 | 1 |
| int8 | 64 | 32 | 1048576 | 1048576 | 0 |
| int8 | 4096 | 32 | 67108864 | 67108864 | 0 |

Wrap identifies unsafe historical reference accumulation. K256/1024 here are numerical studies, not new capabilities of ABI-v1. Current ISA-v0.1 tiling is a separate experiment.

## Full 8x16 kernel storage budget: two slots, two lanes

| Mode | Data scratchpad B | Host quantization parameters B | Accumulator registers B |
| --- | ---: | ---: | ---: |
| int4_acc16 | 112 | 40 | 4 |
| int4_acc32 | 144 | 40 | 8 |
| int8_acc32 | 224 | 40 | 8 |
| fp16_acc32_out32 | 384 | 0 | 8 |
| bf16_acc32_out32 | 384 | 0 | 8 |
| fp16_acc32_out16 | 352 | 0 | 8 |
| bf16_acc32_out16 | 352 | 0 | 8 |
| fp32_ordered | 704 | 0 | 8 |

Budgets include lane padding/output alignment; quantization parameters stay on the host. Control, ports and pipelines are extra. These are not synthesis areas or INT8/FP16/BF16 frequency promises.
Historical bits=16 remains an integer width. Other dtypes exist in explicit NumericMode references; future hardware formats require complete dtype, rounding, scale, FTZ, exception and overflow semantics.
