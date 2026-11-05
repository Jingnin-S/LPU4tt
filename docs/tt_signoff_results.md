# SKY130A macro qualification results

Recorded on 2026-11-04. The independent strict collector reports passed=true
and failures=[], with exit status zero, for the specific validate64_eco15
macro. All builds, simulations and physical checks ran on the server.
Whole-chip power/clock integration, board frequency and shuttle acceptance are
outside this result. There is no silicon measurement.

## Qualified configuration

| Item | Result |
| --- | --- |
| Arithmetic / ISA | W4A4, per-MAC wrap32, raw32/final SAT16, LOAD/STORE/MVM/END |
| Profile | 64B register scratchpad, four synchronous-read banks, one lane, one byte write/clock |
| Technology / tile | SKY130A, 4x2, PDK 0536d02d875c8f67dd7cca3902ac457e62f20005 |
| Clock | 10ns / 100MHz internal macro constraint |
| Worst setup / hold | +1.0945573646ns / +0.1075878861ns |
| Slew / capacitance | Zero violations in nine corners; slew limit 0.75ns |
| Geometry / connectivity | Route/Magic DRC, LVS, antenna, illegal overlaps and critical disconnects: zero |
| TT precheck | Pinned official independent checks, 15/15 |
| Gates | Pin-only arithmetic plus 67 records / 137327 complete cycles |
| Area | Core ~149183um2, die ~154113um2, standard cells ~79714um2, utilization 53.43% |

128B/two-lane and other profiles do not inherit this physical qualification.
INT8/FP16/BF16 remain software experiments.

## Extracted nine-corner STA

| Corner | Setup ns | Hold ns | Slew violations | Cap violations |
| --- | ---: | ---: | ---: | ---: |
| nom_tt_025C_1v80 | 3.471372 | 0.332900 | 0 | 0 |
| nom_ss_100C_1v60 | 1.318800 | 0.554091 | 0 | 0 |
| nom_ff_n40C_1v95 | 4.220212 | 0.109568 | 0 | 0 |
| min_tt_025C_1v80 | 3.559974 | 0.328006 | 0 | 0 |
| min_ss_100C_1v60 | 1.590118 | 0.619636 | 0 | 0 |
| min_ff_n40C_1v95 | 4.280512 | 0.107588 | 0 | 0 |
| max_tt_025C_1v80 | 3.374656 | 0.339678 | 0 | 0 |
| max_ss_100C_1v60 | 1.094557 | 0.477402 | 0 | 0 |
| max_ff_n40C_1v95 | 4.148123 | 0.112225 | 0 | 0 |

The fallback SDC uses 2ns I/O delay, 33.442fF output load, 0.25ns clock
uncertainty and 5% derate. It does not qualify external GPIO at 100MHz.
There are 76 retained max-fanout diagnostics per corner, mostly CTS leaves
with some signal nets; not every diagnostic is zero. Disabled KLayout.DRC/XOR
checks are not represented as executed. TT precheck independently runs its
applicable checks. Missing chip-level IR-drop source modeling is not whole-chip
power qualification.

## Source binding and reproducibility

| Source | SHA256 |
| --- | --- |
| tinylpu_device_memory.sv | d8ba9b8476fa39c75ff73b94a09b3eeb598e1b4115c517bc5073397ae7b78ec2 |
| tinylpu_device.sv | 9fb9d085a29c4ee30b046583b4a6332206dbe13689f40644f4ae78160d5bd246 |
| tt_um_tinylpu_v1.sv | 7c55235ca496cb702e9afc19aab99af1aaac038c043b1e83a4249a198260ca7c |

GDS SHA256: e4d659cc48c178a87ef8a9208e063639e1cb4a49575dd6a993b7d2f29c230b82.
PNL SHA256: b4b6ac66ad8762bda36dd5ca712a4e42165ca8af84f2e28f22c395c9d03c70b0.
The strict reference report also freezes configuration, flow driver, ECO Tcl and
checkpoint hashes. The local buffer ECO preserved RTL/ISA/cycles and all
Classic flow gates; the default unmodified official action alone does not
reproduce this qualified GDS. Failed automatic clock gating was not used.

The source repository contains compact qualification evidence, not the GDS,
mapped netlist, PDK or full physical work directory:
[strict report](../reports/reference/tt_validate64_eco15_signoff.json),
[gate cycles](../reports/reference/device_gate_vectors_validate64_eco15_summary.json),
[precheck](../reports/reference/tt_precheck_validate64_eco15_summary.json),
[source hashes](../reports/reference/source_hashes.json).
Tool/cache locks are in physical/repro.

Current RTL matches the C++ model over 402 records/791566 cycles, with strict
lint and independent arithmetic testing. SGLang integration is a CPU Linear
adapter into DeviceRuntime, not SRT service or a physical driver.
The qualified one-lane M8/K64/batch8 model takes 10887 clocks: 108.87us at a
no-stall 100MHz internal clock, not measured board throughput.
