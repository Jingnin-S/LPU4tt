# Precision comparison and integer-path decision

This numerical study contains seven graphs, seven distributions, 16 inputs per
fixture, 49 fixtures and 539 mode/quantization rows. There are 84 independent
historical INT4/ACC16 layer checks. The current base device uses INT4/ACC32;
INT8/FP16/BF16 remain reference/model experiments.

## Synthetic normal-distribution comparison

Each value is the median graph NRMSE versus the same PyTorch FP32 source.
Integer rows use dynamic per-frame A and per-output-channel W scales.
Floating rows accumulate and output FP32.

| Mode | Normal median NRMSE | Small-input median NRMSE | Fixtures with nonfinite outputs |
| --- | ---: | ---: | ---: |
| INT4 / ACC16 | 0.142152 | 0.164274 | 0 |
| INT8 / ACC32 | 0.00871279 | 0.00849019 | 0 |
| FP16 / ACC32 / OUT32 | 0.000304604 | 1.0 | 8 |
| BF16 / ACC32 / OUT32 | 0.00230954 | 0.00233078 | 0 |

NRMSE is relative output error, not task-accuracy loss. FP16 loses ~1e-9 inputs
and can overflow ~1e5 values or MLP intermediates. BF16 has wider range but lower
significand precision. Storing outputs as FP16 increases nonfinite fixtures
from 8 to 14. Output dtype therefore matters separately.

ACC32 does not recover information lost by INT4 input quantization. Its role is
safe longer-K accumulation and explicit partial sums. INT4 K512 and INT8 K2
already exceed signed INT16 at positive extrema.

Under deliberately shifted outliers, static calibration has median NRMSE near
one, while dynamic A quantization gives approximately 0.0931/0.00494 for INT4/INT8.
This demonstrates sensitivity to calibration distributions, not a universal
ranking of real-model calibration methods. Dynamic host work is additional.

## Storage and hardware implications

A full 8x16, two-slot, two-lane data budget is 112B for INT4/ACC16/OUT16,
144B for INT4/ACC32/OUT32, 224B for INT8/ACC32/OUT32, 352B for FP16/BF16
with FP32 accumulation and 16-bit output, or 384B with FP32 output.
Host scales and internal registers are extra. These are data budgets, not area.

INT4 was selected for the bounded Tiny Tapeout prototype budget. These results
do not establish that W4A4 is optimal for pretrained LLMs. The experiment has
no real model/calibration set, perplexity or task-accuracy claim.
FP references use sequential non-FMA FP32 arithmetic; a different hardware
reduction/FTZ policy requires new checks.

See [contract](precision_experiment_spec.md), [numeric capabilities](isa_numeric_contract.md),
and the [report index](../reports/README.md) for full experiment reproduction.
