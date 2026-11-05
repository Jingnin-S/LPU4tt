# ISA convergence decisions

ISA v0.1 converged on signed INT4 products, ACC32, explicit raw32/SAT16 output,
LOAD/STORE/MVM/END and mixed 4B/8B headers. Encoding is unchanged in this release.
The public wrapper is now 64B/one lane; the earlier 128B/two-lane exploration
remains useful for capacity/throughput comparison.

## Executed experiment scope

- Nine shape fixtures with M=1/3/7/8 and K=1/16/17/64/65/256/1024/4096.
- 1344 main instruction configurations and 13108 tile-cost candidates.
- 336 frequency/link/dispatch scenarios, 10 legacy checks, 31 diagnostics,
  two host-quantized MLP graphs and five controlled ISA ablations.
- Six real torch.export/Relax models, two encodings and 16 inputs each:
  192 frontend inputs.

Independent C++ execution checks numeric output, cycles, MACs and actual byte
counts against Python calculations. INT4/INT8 main sweeps share small integer
inputs to isolate storage/transport; separate diagnostics cover full-range INT8.
These results belong to the ideal-port ISA model, not full-device timing.

## Controlled decisions

The following is a conservative serial scenario at 100MHz and 10MB/s, with
zero per-command host cost; it is not silicon measurement.

| Comparison | Fixed workload | Result | Decision |
| --- | --- | --- | --- |
| Fixed64 -> mixed headers | M8K16, batch8, raw32, 128B, same schedule | Commands 208 -> 136B; 68.42 -> 61.22us | 4B simple headers, 8B MVM |
| Unrolled lane groups -> row-block MVM | Same data/addresses/core costs | Commands 328 -> 136B; 80.66 -> 61.22us | MVM encodes 1..16 rows |
| Reload W -> explicit residency | Same M8K16 inputs | LOAD 576 -> 128B; 111.13 -> 61.22us | Compiler owns LOAD placement |
| Host K partial sums -> ACCUM | M8K256, batch8, same row/K tiling | STORE 1024 -> 256B; 1748.89 -> 1632.73us | Read raw32 seeds in device |
| raw32 -> final SAT16 | M8K16, no clipping in this case | STORE 256 -> 128B; 61.22 -> 47.14us | Separate accumulator/output width |

No descriptor RAM or implicit setup is assumed. LOOP, asynchronous DMA,
WAIT/EVENT and hardware graph/KV management are outside this base ISA.

## Range and capacity

INT4 K512 can produce 32768; ACC16 wraps to -32768, while ACC32 retains it.
INT8 reaches the same boundary at K2. Cross-block cancellation tests retain
raw32 intermediate values and obtain 4096 after cancellation instead of
incorrectly clipping intermediate partial sums.

M8K16/raw32 needs W64 + A8 + Y32 = 104B before bank padding. 128B crosses that
residency threshold. SAT16 reduces the single-slot lower bound to 88B; a 96B
ideal-port profile can help, subject to explicit clipping/range semantics.
INT8/raw32 needs 176B. Larger capacity helps long-K reuse, but requires its own
physical area and clock assessment.

At fixed 10MB/s, changing the modeled clock from 100 to 200MHz changes the
M8K16/batch8/raw32 estimate from 61.22 to 56.61us (about 7.5%). At fixed
100MHz, 10 -> 50MB/s changes it to 19.62us. This motivates optimizing traffic
and reuse alongside core timing. First-frame latency and cold-batch throughput
must be reported separately.

See [main data](../reports/isa/comparison.csv),
[search](../reports/isa/tiling_search.csv),
[scenarios](../reports/isa/performance_factors.csv),
[diagnostics](../reports/isa/diagnostics.json),
[frontend summary](../reports/isa/frontend/summary.json),
and [ablations](../reports/isa/ablations/summary.json).
Reproduction commands are in the [modeling guide](modeling.md).
