# Physical development summary

The final qualified result is [validate64_eco15](tt_signoff_results.md).
This summary preserves the engineering lessons from the earlier experiments;
historical candidates do not inherit the final result.

The server used rootless Podman, LibreLane 3.0.14 and pinned SKY130 PDK
0536d02d875c8f67dd7cca3902ac457e62f20005, with pinned Tiny Tapeout template
and support tools. The official example produced GDS, RTL/gate tests and
15/15 precheck before LPU qualification. A manually selected matching PDK
was needed because image and precheck defaults differed.

| Candidate family | Observation |
| --- | --- |
| Original 128B/two-lane, 10ns | Routing failed to converge |
| fastarb 128B/two-lane, 20ns | Geometry/gate checks passed, but SS setup/electrical checks failed |
| fastarb 128B/two-lane, 10ns | Geometry passed; SS setup -5.2915ns, slew/cap failures |
| Broad MCMM/repair trials | Some runs stopped for sustained memory growth or timeout; not passes |
| regptr/onehot/speed studies | Area/control experiments still had timing/electrical failures |
| validate64 initial | Geometry, gate and precheck passed; timing/electrical gates still failed |
| Automatic clock gating | Failed synchronous-clear behavior; disabled and not used |
| Smaller 3x2 tile | Lower die area did not resolve electrical/timing failures |
| Strong buffering and guard | Positive timing; residual slew violations remained |
| ECO14 | Positive timing, ten slew violations remained |
| ECO15 | Final nine-corner timing/electrical, geometry, gate and precheck passed |

The final local buffer ECO used real extracted driver/sink evidence. It retained
10ns/0.75ns constraints, RTL/ISA/cycles, Classic flow steps and gating; subsequent
placement, routing, extraction and signoff ran normally. Default official-action
output without that ECO is not the qualified GDS.

The selected 4x2 result uses about 79714um2 standard-cell area at 53.43%
reported utilization. Capacity and lane count therefore remain physical trade-offs.
Generic 45nm probe areas and timing are historical comparisons, not SKY130 data.

Final reference JSON retains 76 max-fanout diagnostics separately. Whole-chip
power/clock integration, silicon measurements and board frequency qualification
remain outside this macro evidence. Full raw server work directories, PDKs,
vendor caches and failed-flow logs are not distributed in this source repository.
Locked environment reproduction is in physical/repro.
