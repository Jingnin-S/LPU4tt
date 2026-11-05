# Tiny-LPU physical-flow bootstrap

This directory contains MIT-licensed project-owned bootstrap metadata and a
small shell checker.  It does not contain a PDK, container layers, vendor
source, mapped netlists, or private machine paths.  Upstream template/action
trees keep their own licenses; the exact source commits are recorded in
`versions.lock`.

## Locked inputs

`versions.lock` records the observed LibreLane 3.0.14 image ID and registry
digests, OpenROAD build commit, Sky130A/open_pdks commit, Tiny Tapeout
template/action/support-tool commits, Ciel version, and the versions of the
container Python, cocotb, layout, Cairo, NumPy, and Yosys packages.  The
server validation used the x86_64 image digest in `LIBRELANE_IMAGE_REF`.

The complete-device tests use a separate `.venv-cocotb-container` with
cocotb 2.0.1 on the image's Python 3.13.9, Icarus 13.0, and Verilator 5.044.
The larger layout/bootstrap venv has cocotb 2.1.0. These are separate
environments; do not mix one venv's makefiles/VPI library with the other.
The C++/compiler/ESL host environment remains Python 3.11.16 and GCC 11.5.0.
The observed full-device versions are recorded separately in `versions.lock`.

The project cache is expected to contain these relative resources when they
are available:

```text
physical/pdk/sky130A
physical/vendor/{ttsky-verilog-template,tt-gds-action,tt-support-tools,ciel-action}
physical/container-venv
```

No PDK or image is downloaded by `--check`.

## Commands

Run from the project root on Linux:

```sh
physical/repro/bootstrap_tt_env.sh --check
```

`--check` is read-only with respect to the project and uses isolated rootless
Podman storage.  It verifies the vendor commits, PDK `SOURCES`, image ID and
both observed registry digests, LibreLane/OpenROAD versions, and a container
import probe for Python, cocotb, gdstk, KLayout, NumPy, CairoSVG, and
Yosys.

To prepare a *new, empty* cache without touching an existing project or work
directory:

```sh
physical/repro/bootstrap_tt_env.sh --bootstrap \
  --cache-root /absolute/path/tinylpu-tt-cache
```

This writes only `env.sh`, `podman-root/`, and `podman-runroot/` below the
newly-created cache.  Add `--pull-image` only when an intentional network
download is wanted; it pulls the immutable `LIBRELANE_IMAGE_REF` and tags it
with the locked version name.  The cold-cache pull path was not exercised in
the project validation; the existing server cache and all checks were
validated.  PDK acquisition/activation is intentionally not automated here,
so it cannot silently replace a pinned PDK.

For an isolated cache that has already been populated, load its paths before
checking it:

```sh
. /absolute/path/tinylpu-tt-cache/env.sh
physical/repro/bootstrap_tt_env.sh --check
```

Useful non-invasive overrides are `TINYLPU_PROJECT_ROOT`,
`TINYLPU_PDK_ROOT`, `TINYLPU_CIEL_BIN`, `TINYLPU_PODMAN_ROOT`, and
`TINYLPU_PODMAN_RUNROOT`.  The script rejects a non-absolute or already
existing bootstrap cache and refuses paths under `physical/work`.

## Container runtime note

## Reproducing the LPU candidate flow

The following is the complete *manual* command chain for a server that already
has the locked PDK, LibreLane image, support tools, and the separate physical
container/cocotb environments in its cache.  It is an execution recipe, not a
claim that a cold cache or a new GDS was reproduced by this source archive.
Run the host model/vector commands with a Python 3.11 interpreter and the
compiler libraries built from this checkout; this keeps Nix/Python 3.13 from
loading a host-built C++ shared object.

```sh
PY=/absolute/path/to/python3.11
export PYTHONPATH="$PWD/compiler"
export LD_LIBRARY_PATH="$PWD/build_esl"

# The validated cache/venvs must be visible below this checkout when mounted
# as /work; external symlinks require their own explicit volume mounts.
TINYLPU_RUN_ROOT="$PWD"
run_ll() {
  podman --root /tmp/tinylpu-tt-podman-overlay \
    --runroot /tmp/tinylpu-tt-podman-overlay-run \
    --storage-driver=overlay --storage-opt overlay.ignore_chown_errors=true \
    run --rm --network=none --security-opt label=disable \
    -v "$TINYLPU_RUN_ROOT:/work" -w /work \
    ghcr.io/librelane/librelane:3.0.14 "$@"
}

# Build the host C++ libraries, then create the six independent vector sets.
cmake -S . -B build_esl -DCMAKE_BUILD_TYPE=Debug
cmake --build build_esl --parallel 2
for profile in 64:1 64:2 128:1 128:2 256:1 256:2; do
  capacity=${profile%:*}; lanes=${profile#*:}
  "$PY" scripts/run_device_vectors.py \
    --capacity "$capacity" --lanes "$lanes" \
    --output "reports/device_mvmvalidate_vectors/c${capacity}_l${lanes}"
done
"$PY" scripts/run_device_performance.py \
  --output reports/device_performance

# Inside the locked LibreLane image, consume those host-generated vectors and
# create the C++/RTL model receipt used by the gate-vector checker.  Supplying
# --vectors-root is intentional: the Nix Python must not load host-built .so
# files while regenerating the vectors.
run_ll python3 scripts/run_device_rtl.py --simulator iverilog \
  --vectors-root reports/device_mvmvalidate_vectors \
  --output reports/device_mvmvalidate

# Use the public root itself as the Tiny Tapeout source project.  The guard
# pass uses the 64B/1-lane profile, 10 ns clock, 4x2 tile, and 50/50/3 GRT
# margins; it creates physical/work/validate64_guard.
"$PY" scripts/run_tt_candidate.py --base-directory . \
  --name validate64_guard --capacity 64 --lanes 1 --clock-period 10 \
  --ss-corner --post-grt --speed-cells --strong-buffers \
  --grt-slew-margin 50 --grt-cap-margin 50 --grt-setup-margin 3

# Start a new ECO15 candidate from the completed guard timing checkpoint.
"$PY" scripts/resume_tt_late_repair.py \
  --base validate64_guard --name validate64_eco15 \
  --slew-margin 60 --cap-margin 50 \
  --eco-pins-file configs/tt_guard_slew_eco15_pins.json

# Fresh routed pin-only GL and complete routed cycle-vector GL receipts.
run_ll /work/.venv-cocotb-container/bin/python scripts/run_device_gate.py \
  --candidate physical/work/validate64_eco15 \
  --output reports/device_gate_validate64_eco15 --stage routed --unit-delay 1
run_ll python3 scripts/run_device_gate_vectors.py \
  --candidate physical/work/validate64_eco15 \
  --output reports/device_gate_vectors_validate64_eco15 \
  --vectors-root reports/device_mvmvalidate_vectors \
  --model-report reports/device_mvmvalidate/summary.json

# Prepare the precheck envelope with the exact GDS/LEF/PNL and matching info.
# run_tt_precheck.py expects .gds, .lef, and a sibling .v basename.
mkdir -p physical/work/parent_checks/validate64_eco15_input
cp physical/work/validate64_eco15/runs/wokwi/final/gds/tt_um_tinylpu_v1.gds \
   physical/work/parent_checks/validate64_eco15_input/tt_um_tinylpu_v1.gds
cp physical/work/validate64_eco15/runs/wokwi/final/lef/tt_um_tinylpu_v1.lef \
   physical/work/parent_checks/validate64_eco15_input/tt_um_tinylpu_v1.lef
cp physical/work/validate64_eco15/runs/wokwi/final/pnl/tt_um_tinylpu_v1.pnl.v \
   physical/work/parent_checks/validate64_eco15_input/tt_um_tinylpu_v1.v
cp physical/work/validate64_eco15/info.yaml \
   physical/work/parent_checks/validate64_eco15_input/info.yaml
"$PY" scripts/run_tt_precheck.py \
  --gds physical/work/parent_checks/validate64_eco15_input/tt_um_tinylpu_v1.gds \
  --label validate64_eco15

# Bind physical, GL, vector, and exact-RTL receipts; --strict is the gate.
"$PY" scripts/collect_tt_signoff.py \
  --candidate physical/work/validate64_eco15 \
  --precheck physical/work/parent_checks/validate64_eco15/summary.json \
  --gate reports/device_gate_validate64_eco15/summary.json \
  --gate-vectors reports/device_gate_vectors_validate64_eco15/summary.json \
  --output reports/validation/tt_validate64_eco15_signoff.json --strict
```

The default upstream `tt-gds-action` parent flow is not substituted for this
receipt and is not reported as an ECO15 signoff.  The strict collector also
retains the separate max-fanout diagnostics; a zero DRC/LVS/antenna/slew/cap
qualification does not mean every diagnostic count is zero.

The LibreLane image uses Nix-provided Python and C/C++ libraries.  The
validated probe mounts the project `physical/container-venv` read-only and
constructs `LD_LIBRARY_PATH` from the cocotb bundled libraries plus the
container's `/nix/store` directories containing `lib*.so*`.  This is why the
KLayout/gdstk/Cairo imports work without changing the host loader or installing
system packages.  The same isolated environment was used for the official
example harden, RTL/GL tests, and precheck runs.

## Reproducibility boundary

The checked-in script proves the versions and cache layout of an already
prepared environment.  It does not claim that a cold PDK/image download,
license-server access, or a complete LPU signoff is reproducible offline.
Physical-flow results must retain their actual corner/constraint status; the
Sky130A PDK and LibreLane/vendor licenses are not relicensed by these files.
