# Contributing

Use Python3.11 and the pinned CPU integration requirements. Run
`PYTHON=python3.11 bash scripts/check_release.sh` after C++/compiler changes.
Keep numerical, resource-timing and hardware claims separate. ISA changes need
explicit encoding/arithmetic/error contracts and independent reference tests.
Do not commit build/venv directories, model weights, machine-specific logs,
credentials or third-party source. RTL changes need matching model/vectors.
