# Validation and evidence

The v0.2.1 publication tree was built and checked on the Linux server.
No model arithmetic, ISA bytes or qualified RTL were changed for documentation
consolidation.

## Software and model checks

The clean-tree check_release.sh run passed 10/10 CTest targets, 24 analysis tests,
69 software tests, dependency consistency and the real
torch.export -> Relax -> fused VM callback -> ISA C++ smoke test.

The release experiments include 72 complete-device configurations with independent
numeric and phase-count agreement, 1344 ISA main configurations, 192 real
frontend inputs and five controlled ISA ablations. See the
[report index](reports/README.md) for fresh compact results and provenance.

Historical resource studies cover 120 executed configurations, 4596 tile/order
candidates and 2160 clock/link/dispatch scenarios. Real VM composition covered
42 combinations and 378 input frames. These resource cycles intentionally differ
from the complete-device FSM contract.

## RTL and physical evidence

The unchanged v0.2.0 qualified sources retain six-profile RTL results:
402 records, 791566 cycles, Icarus simulation and zero-warning Verilator lint.
Independent pin-only cocotb and complete routed gate vectors are separately
recorded. The 64B/one-lane SKY130A macro has positive setup/hold in nine corners,
zero cap/slew and DRC/LVS/antenna errors, and 15/15 TT precheck.
Max-fanout diagnostics remain explicit. See
[qualification](docs/tt_signoff_results.md) and compact reference JSON.

No physical flow rerun is implied by a documentation-only release. Source
hashes bind the previous evidence. Other profiles do not inherit that result.

## Upstream integration and accuracy limits

The real installed SGLang 0.5.22 ReplicatedLinear experiment passed six CPU
shapes/twelve calls with post-load, bias and skip-bias behavior through
DeviceRuntime. It is module-level software-to-model integration, not full SRT,
distributed serving, KV management, token generation or a hardware GPIO driver.

Synthetic floating-source Linear shapes had roughly 0.107-0.199 NRMSE versus
FP32 under dynamic INT4 quantization. Exact quantized-oracle agreement is not
pretrained-model accuracy. INT8/FP16/BF16 remain numerical references.

GitHub Actions runs the reproducible software integration check after upload;
server checks and GitHub-hosted results are reported separately.
