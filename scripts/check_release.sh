#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tlpu_python="${PYTHON:-python3.11}"
cmake -S . -B build_esl -DCMAKE_BUILD_TYPE=Debug
cmake --build build_esl --parallel 2
ctest --test-dir build_esl --output-on-failure
"$tlpu_python" -m unittest discover -s tests -p 'test*analysis.py' -q
"$tlpu_python" -m pytest tests/software -q
"$tlpu_python" -m pip check
"$tlpu_python" scripts/run_release_smoke.py
