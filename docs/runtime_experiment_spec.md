# Historical ABI-v1 VM and end-to-end runtime experiment

This contract records the earlier batch-one ABI-v1 integration. Current
ISA-v0.1 composition and DeviceRuntime are described in compiler.md and the
device contract; their timing is not substituted into this older experiment.

The four original models use strict INT4 values carried in float32 tensors,
constant weights and ACC16 outputs. Shape, dtype, finiteness, integrality and
range are checked. MLP host quantization remains
clip(floor(relu(x)*0.75+0.5),0,7).

The path is torch.export/Relax partition -> Tiny IR -> executable Relax VM
glue -> per-GEMV packed callback -> in-process C++ shared library -> DseCore.
Host quantization remains an actual Relax CPU operation. This is supported-
subset explicit lowering, not a general TVM RunCodegen plugin.

The runtime C ABI creates, runs and destroys validated plan/weight handles.
State resets on each run; no cross-run free residency is assumed.
Exceptions become errors, strings follow header lifetimes, Python serializes
session access and checks hashes. Invalid runs clear stale reports.

Report three separate time classes:

1. Export/compile/VM-build/load wall time.
2. Warm software VM/callback/C++ simulation wall time, including FFI costs.
3. Parameterized sequential hardware scenarios built from actual event order.

Scenario configuration scans 50/100/200MHz and 1/10/50MB/s. Assumed 8-byte
device commands are analysis parameters, not the 32-byte host plan instruction
or current mixed-width ISA. Count command bytes, setup, A/W/Y transfers,
per-plan-instruction host dispatch, per-partition VM boundary, packing,
unpacking and quantization without double counting callback internals.
Each invocation pays its actual LOAD_W bytes. No double-buffer overlap is assumed.

Validate at least 32 inputs per original model against PyTorch and independent
C++ arithmetic. Check callback counts, cycles/MAC/bytes, host quantization IR,
input variation, repeated runs, corrupted artifacts, error recovery, event
time sums and clock/link/host-cost sensitivity.
