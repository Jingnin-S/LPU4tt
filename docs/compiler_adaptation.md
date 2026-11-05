# Compiler adaptation and debugging LPU composition

The executable chain is now torch.export → actual Relax partition → Tiny IR DAG
→ target/layout/composition lowering → actual Relax VM → ISA-v0.1 C++ execution.
The original ABI-v1 VM remains available. This backend is an explicit supported-
subset lowering, not a general TVM RunCodegen plugin.

`import_model(..., multi_output=True, allow_batch=True)` opts into tuple DAG
outputs and batches; default old behavior stays unchanged. `CompositionVm` accepts
target `Profile`, memory `Resources`, `storage='packed'|'bank4'` and `fuse_qkv`.
Siblings can fuse only when they share the exact same source. Host Q34 remains a
native Relax CPU operator and a hard numeric/dependency boundary.

```python
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.composition_vm import CompositionVm
from tinylpu_compiler.isa import Profile
from tinylpu_compiler.architecture import Resources

nodes, metadata = import_model(model, example_args, directory / "frontend",
                               multi_output=True, allow_batch=True)
with CompositionVm(nodes, directory / "vm",
                   "build_esl/libtinylpu_isa.so",
                   "build_esl/libtinylpu_architecture.so",
                   frames=8, profile=Profile(128, 2, 3, 32, 1),
                   resources=Resources(4, 1, 1, 1),
                   storage="bank4", fuse_qkv=True) as program:
    outputs = program.run(float32_exact_int4_array)
    trace = program.last_reports
```

Debug in this order:

1. `frontend/imported_relax.ir.txt` and `partitioned_relax.ir.txt`: check actual
   upstream partition and whether an unsupported op appears.
2. `graph_debug.json`: inspect host/device placement and source-node dependencies.
3. `partition_*/debug.json`: inspect fused source mappings, row/K search, A/W/Y
   addresses, bank padding, logical disassembly and actual byte totals.
4. `commands.bin`/`weights.bin`/manifest: identify the precise executable kernel;
   runtime A remains separate, never precomputed into outputs.
5. `last_reports`: compare numeric result/input/output hashes and resource timing.
   A numeric mismatch is separate from a bank-conflict or performance mismatch.
6. Compare presets and fusion flags in `run_composition_experiments.py`. Report
   callback count and host costs along with device timing. Bank padding can make
   fusion less useful by changing the maximum resident row tile.

Invalid input clears per-run reports before rejection. A closed VM or cross-thread
use is rejected; registry callbacks are removed when the VM closes. Compiled
weights and shape/schedule are reused for multiple runtime inputs.

## SGLang Linear bridge

`sglang_adapter.attach_linear(layer, isa_library, architecture_library)` explicitly
wraps an already loaded layer's `quant_method`. It does not modify all SGLang
modules or register a new device. Supported CPU float32 inference shapes use
per-frame activation/per-row weight symmetric RNE INT4 quantization, raw32 device
output, host dequantization and optional bias. Unsupported device/dtype/shape or
gradient calls delegate to the original method with a recorded reason. Weight
content hashes prevent stale cache reuse after mutation. This bridge performs
approximate quantization; it is not a bit-exact replacement for original FP32.

The integration driver downloads the fixed upstream
[SGLang Linear source](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/linear.py),
extracts the actual ReplicatedLinear.forward function and executes that callsite
with the interposer. Both bias and skip-bias-add contracts are tested, as is the
quantized integer oracle. A second integration test imports the installed
SGLang0.5.22 package, normally constructs ReplicatedLinear, loads weights, runs
the real post-load hook, attaches the adapter and executes forward on six CPU
shapes. Both bias modes and the integer/dequant oracle pass. AMX-packed CPU
weights are explicitly delegated to the original method rather than interpreted
as a dense matrix. Tensor parallel collectives, SRT scheduling, KV cache,
serving and generation are outside these module-level tests.

Run the optional installed-package test in a SGLang CPU environment:

```sh
SGLANG_USE_CPU_ENGINE=1 python scripts/run_sglang_layer_experiment.py
```

Keep that environment separate from the Relax compiler: SGLang0.5.22 declares
TVM FFI0.1.11, while the compiler uses0.1.15.post2. Actual installed-layer tests
used0.1.11; the serving adapter imports no TVM/Relax and uses only lightweight
plan codegen, NumPy/Torch and the independent C ABI. Compiler artifacts and
C ABI libraries are the deployment boundary, not a merged dependency environment.

For the six small synthetic floating-source shapes, NRMSE versus FP32 was about
0.107–0.199; exact agreement with quantized arithmetic does not establish LLM
quality. Existing INT8/FP16/BF16 experiments are references for future dtype
decisions, not implemented base-device support.
