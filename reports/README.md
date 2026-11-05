# Report index

Compact results are published with source and reproduction commands.
Large traces, frontend artifacts, environments and raw machine logs are
generated on demand.

| Study | Result files | Reproduce |
| --- | --- | --- |
| ISA execution | isa/comparison.csv, summary.json, diagnostics.json | run_isa_experiments.py |
| ISA frontend | isa/frontend/summary.json | run_isa_frontend_experiments.py |
| ISA controlled ablations | isa/ablations/summary.json | run_isa_ablations.py |
| Full-device cycles | device_performance/comparison.csv and summary.json | run_device_performance.py |
| Three-template ESL | esl/summary.csv and summary.json | analyze_esl.py |
| Common operators | operators/summary.csv and run.json | analyze_operators.py |
| Numerical precision | precision/summary.csv and results.json | run_precision_experiments.py |
| Existing resource/composition | reference/resource_*, composition_* | run_architecture_experiments.py / run_composition_experiments.py |
| Qualified RTL/physical evidence | reference/*summary.json and tt_validate64_eco15_signoff.json | See validation.md / physical/repro |
| Publication checks | reference/lpu4tt_v0_2_1_checks.json | check_release.sh and publication audit |

The current source hashes bind retained v0.2.0 RTL/physical receipts.
Fresh software experiments are identified by the v0.2.1 check record.
No hardware measurements are inferred from modeled MHz or CPU runtime.
