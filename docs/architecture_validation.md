# Architecture and compiler validation

> Historical resource/probe study. Its 128B/two-lane recommendation and
> implementation-to-do statements describe that stage, not the current release.
> Current device and physical status: [architecture](../architecture.md),
> [device contract](device_rtl_contract.md), [qualification](tt_signoff_results.md).

All experiments and builds for this stage execute on a Linux server. The local
workspace is a synchronized delivery mirror. This report distinguishes executed
software checks from proposed full-chip RTL and physical signoff.

## Completed experiments

| Experiment | Executed coverage | Result |
| --- | --- | --- |
| Port/bank/capacity compiler selection |120 numerical ISA configurations,4596 tile/order candidates|PASS |
| Frequency/link/dispatch scenarios |2160 modeled scenarios|generated from actual byte/resource counters |
| New Relax VM backend/composition |42 combinations,378 validated input frames|PASS |
| SGLang callsite bridge |6 CPU shapes,actual pinned forward AST,bias and skip-bias-add|quantized oracle PASS |
| Installed SGLang0.5.22 module |6 normally constructed ReplicatedLinear shapes,post-load hook,12 offload calls|PASS |
| New Python backend tests |7 targeted tests|PASS |
| Clean-checkout real frontend smoke |torch.export→Relax→fused callback→actual ISA C++|PASS |

Existing ISA-v0.1 evidence remains1344 executed configurations,192 real-frontend
inputs,5 controlled ISA ablations and independent integer/floating references.
The new resource analyzer counts registered memory reads, bank conflicts and byte
copies. Its cycles are deliberately different from the old ideal-port model.

M8K16,batch8,raw32,100MHz/10MB/s: flat3-read128B=64.74us; bank4 packed=74.98us;
bank4 padded=66.50us. Bank padding removed1024 conflict cycles while increasing
real payload/storage. Moving to4 lanes gave66.02us, a small benefit in this case.

QKV,batch8,1us per host partition: flat3-read128B separate151.14us versus fused
127.83us; bank4-padded128B separate156.42us versus fused153.60us. Fusion is a
target/layout decision, not an unconditional optimization.

The six floating-source SGLang Linear shapes had about0.107–0.199 NRMSE versus
FP32 under dynamic INT4 quantization. Passing the quantized arithmetic oracle does
not establish real-model quality. The actual package Linear module was also
tested; full SRT serving, distributed tensor parallelism, KV management and
token generation were not exercised. Compiler and SGLang environments use their
respective TVM FFI versions0.1.15.post2 and0.1.11, with the C ABI as boundary.

## Meaning of the hardware evidence

Memory and INT4 MAC probes provide selected RTL simulation and generic synthesis
evidence. They do not include the complete byte controller, gather routing,
validity bitmap, output saturation/control, pad/clock/reset network or routing.
Mapping used gscl45nm typical1.1V/27C, with no routing parasitics. The corrected
native one-read-per-bank arrays and DSP_DEPTH3 MAC were simulated with VCS;
36/36 generic synthesis cases and21/21 mapped cases passed. Fifteen timing checks
covered10/6.667/5ns on selected probes. The 128B flat3-read array reported26620.57
library area units, versus18394.68 for native bank4, about30.9% less. The192B bank4
array reported28326.95, about54% more than128B. These array figures omit gather
routing/controller/validity costs. The two-lane ACC32 DSP_DEPTH3 probe reported
2278.92 area units. There is no target Tiny Tapeout PDK or full-chip STA result.
Generic cells must not be described as micrometers, power, tile fit or Fmax.

The full logical architecture, ports, states, reset and pin proposal are documented
in architecture.md. Full framer/controller RTL and cycle-equivalence under reset,
backpressure and error conditions remain implementation acceptance tasks. A
complete architecture proposal is distinct from a completed physical chip.

## Reproduction and release checks

The separate public staging tree was built from scratch on the server:9/9 CTest,
57 software tests,24 analysis tests,dependency check and the real frontend/VM/
ISA smoke all passed. A Python wheel also built successfully. This clean-tree
check includes unsupported zero-K/bias fallback guards in the opt-in Linear
adapter; it does not require importing the complete SGLang package.

Run `PYTHON=python3.11 bash scripts/check_release.sh` in a fresh tree to build the
C++ targets, execute CTest/analysis/software regressions, check dependencies and
run the real frontend/VM/ISA smoke test. Run the two architecture drivers for full
resource/composition reproduction. The optional RTL probe driver requires the
documented simulator and synthesis tools.

The public tree is curated: original code/tests/configuration, MIT license,
workflow, reproduction scripts, design/debug documentation and compact reference
results. Environments, binary libraries, model weights, full raw machine logs,
private server paths and downloaded third-party source are excluded. A credential
and private-path pattern scan is mandatory before packaging. GitHub Actions
configuration is supplied; the server check does not claim GitHub-hosted execution.
