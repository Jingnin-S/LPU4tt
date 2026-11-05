# Numeric contract and capability boundaries

ISA v0.1 and the released device use W4A4/ACC32. Historical ABI-v1 remains
INT4/ACC16; its integer width fields and host operations retain their meanings.

| Operands | Accumulator / output | Status |
| --- | --- | --- |
| INT4 / INT4 | INT16 / INT16 | Historical ABI-v1; bounded K and capacity |
| INT4 / INT4 | INT32 / raw32 or SAT16 | Current ISA v0.1 and device RTL |
| INT8 / INT8 | INT32 / raw32 or SAT16 | Explicit optional ISA-model experiment |
| FP16 / FP16 | FP32 / FP16 or FP32 | Numerical reference only |
| BF16 / BF16 | FP32 / BF16 or FP32 | Numerical reference only |

A width of 16 does not identify FP16 or BF16. Any future extension must declare
activation, weight, accumulator and output dtypes independently, together with
rounding, saturation/wrap, scale location/granularity, zero-point and exceptions.
Shape, storage and format capability checks are separate.

## Integer behavior

Signed INT4 spans [-8,7]; INT8 spans [-128,127]. Raw outputs are little-endian.
Every current-device MAC wraps at 32 bits. ACCUM reads raw32 seeds. SAT16 only
clamps the final output of a command; intermediate K tiles must retain raw32.
Arbitrary seeds or unlimited ACCUM sequences need their own range analysis.

The largest positive INT4 product is 64. K64 can produce 4096, while K512 can
produce 32768 and overflow ACC16. INT8 can reach 32768 at K2. Zero-initialized
INT4/INT8 dots with total K <= 4096 fit ACC32.

The historical HOST_RELU_Q remains ReLU, fixed 3/4 scaling, nonnegative
half-up rounding and clipping to [0,7]. New PTQ uses round-to-nearest-even (RNE);
it is a different host operation.

## Quantization and floating references

The numerical experiment stores FP32 scales on the host: per-output-channel W
and either dynamic per-frame A or independently calibrated static A.
Scale storage and arithmetic are not free device resources.

FP16/BF16 references use RNE input conversion, preserve subnormals, and form
each FP32 product followed by a separate sequential FP32 addition in K order.
They do not use FMA. Optional output conversion is separate. A future FMA,
tree reduction or flush-to-zero implementation needs a new validated contract.
NaN payload is unspecified; class, infinity sign, finite encoding and signed
zero are independently checked.

See [ISA](isa_v0.md), [precision contract](precision_experiment_spec.md)
and [precision results](precision_results.md).
