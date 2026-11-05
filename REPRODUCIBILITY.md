# Reproducibility

Use Linux, C++17, CMake >= 3.16 and Python 3.11. requirements.txt pins the
full CPU PyTorch/TVM integration environment; requirements-report.txt contains
optional plotting dependencies. No GPU is required.

~~~sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON=.venv/bin/python bash scripts/check_release.sh
~~~

check_release.sh builds C++, runs CTest, analysis/software checks, pip check and
a real export/Relax/ISA fused-QKV smoke test. [Modeling commands](docs/modeling.md)
reproduce ISA, operator, numeric, resource and complete-device studies.
run_composition_experiments.py fetches a pinned upstream source contract.
The installed SGLang experiment requires its separate compatible environment.

Full generated artifacts can contain absolute local library paths. Compact
published results normalize the repository root to a symbolic marker and omit
private machine logs. Tests execute the original generated artifacts before
publication normalization. No numeric data or model source is changed.

Do not compare ideal-port ISA cycles, explicit resource estimates, complete
device clocks and Python/VM wall time as if they were the same metric.
Template clocks are assumptions; physical results apply only to their bound
source/profile. See validation.md and the qualification report.

physical/repro documents pinned user-level tools and cache-only verification.
Cold PDK/container acquisition and a rerun of the qualified local ECO are
separate from source/model reproduction. No PDK, third-party source cache,
environment, weights or full physical work directory is included.
