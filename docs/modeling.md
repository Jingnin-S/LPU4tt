# Functional and performance modeling

The project keeps independent functional execution and cycle accounting so
that an attractive schedule cannot hide an incorrect numeric result.
All release builds and experiments were executed on the Linux server.

## Model layers

| Layer | Implementation | What it establishes |
| --- | --- | --- |
| Initial ESL | tinylpu_esl | dot4/GEMV behavior, handshakes and phase counts |
| Operator ESL | tinylpu_operators / tinylpu_nonlinear | Integer operator schedules and nonlinear primitive DAGs |
| Numeric reference | tinylpu_precision | INT4/INT8/FP16/BF16 error and range comparison |
| ISA execution | tinylpu_isa | Decoding, runtime scratch values, wrap/clamp, bytes and ideal-port cycles |
| Resource analysis | tinylpu_architecture | Address-derived port/bank conflicts, registered reads and serialized copies |
| Full device | tinylpu_device + device.py | Actual FSM/pins, backpressure, initialization, exact device cycles |
| RTL/gate comparison | run_device_vectors / run_device_rtl | Pin-by-pin agreement with the device model |

Later layers do not silently replace older timing assumptions. Initial operator
models use ACC16 in their explicitly bounded cases. The current ISA/device uses
ACC32. Nonlinear DAGs are resource studies, not opcodes implemented by the RTL.

## Three Tiny Tapeout templates

SKY, IHP and GF snapshots live in configs/templates. Each instantiates the same
behavioral core with its own pinned template/PDK metadata. At equal resource,
clock and workload assumptions, cycle counts must be identical across templates.
Template die area and nominal clock do not predict area, power, tile fit or Fmax.
Only the specific SKY130A 64B macro has the separately recorded physical evidence.

## Reproduce after building

Use the Python 3.11 environment from the root README. Commands below assume
the current directory is the repository root.

~~~sh
.venv/bin/python scripts/analyze_esl.py --model build_esl/tinylpu_esl
.venv/bin/python scripts/analyze_operators.py   --integer-model build_esl/tinylpu_operators   --nonlinear-model build_esl/tinylpu_nonlinear

.venv/bin/python scripts/run_isa_experiments.py   --library build_esl/libtinylpu_isa.so   --legacy-library build_esl/libtinylpu_plan_runtime.so
.venv/bin/python scripts/run_isa_frontend_experiments.py   --library build_esl/libtinylpu_isa.so
.venv/bin/python scripts/run_isa_ablations.py   --library build_esl/libtinylpu_isa.so

.venv/bin/python scripts/run_architecture_experiments.py
.venv/bin/python scripts/run_device_performance.py
.venv/bin/python scripts/run_precision_experiments.py   --numeric-library build_esl/libtinylpu_precision.so   --legacy-library build_esl/libtinylpu_plan_runtime.so
~~~

The optional report plotting dependencies are in requirements-report.txt.
run_composition_experiments.py additionally exercises the real Relax VM and
fetches a pinned SGLang source contract. The installed SGLang experiment needs
its own environment because its TVM FFI requirement differs; see compiler.md.

## Performance accounting

For a banked group and each K element, gather unique A/W byte addresses.
Equal addresses broadcast. The number of request rounds equals the largest
number of distinct requested addresses in any bank. The full-device model adds
all control, registered response, seed, drain, writeback and STORE phases.
The independent Python phase sum must equal executed C++ ticks.

No-stall device time is cycles/fclk. A byte-level device simulation already
includes one handshake cycle per transferred byte; add only extra waits or
model the actual external ready/valid schedule. Do not add an entire second
copy of byte transfer time to that result.

The older conservative ISA/resource scenario model adds serial link time
and host dispatch to internal cycles. It deliberately gives no overlap credit
and is a different abstraction. Software callback/VM wall time includes
simulation, Python and FFI costs and is not hardware latency.

Report workload M/K/batch, numeric profile, output width, capacity, lane count,
layout, tile/order, clock and link assumptions together. Keep first-frame
latency separate from cold-batch throughput; all modeled batch frames arrive
at time zero, with no aggregation wait.

## Reading results and debugging

1. Check inputs and weights against the independent mathematical oracle.
2. Inspect graph placement, commands, disassembly and scratch allocation.
3. Compare byte/MAC counts and bank conflicts across layouts.
4. Compare full-device phase totals against executed C++ ticks.
5. Use RTL vectors for reset/start/ena/error/backpressure boundaries.
6. Apply each configuration's actual physical clock/area evidence only afterward.

Compact CSV/JSON results are indexed in reports/README.md. Full runtime
artifacts, environments, PDKs and private machine logs are excluded from Git.
