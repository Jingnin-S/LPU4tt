# Changelog

## Unreleased - device cycle optimizations

ISA v0.1 encoding and arithmetic are unchanged. The device RTL, the C++ pin/cycle
model and `no_stall_cycles` change together, so cycle counts differ from v0.2.1.

- MVM core: each fetched byte feeds both nibbles to the MAC, with two 4x4
  products per lane. One request round now covers two K elements.
- ACCUM seed: one four-bank read per lane (1 + lanes clocks, was 8 per lane).
- Header checking: LOAD/STORE/END go from the last header byte to CHECK
  (header + 1); MVM uses ENDS and MVM_VALIDATE and faults in group
  initialization (header + 2).
- STORE: reads ahead of the registered TX byte (count + 2 clocks, was 3 per byte).
- Writeback: one write per bank and clock, so one clock per lane.

The 72-configuration device experiment takes 23% to 44% fewer clocks (mean 35%)
with identical plans, wire bytes and results. The 64B/one-lane M8/K64/batch8
case goes from 10887 to 7782 clocks.

Checked with Icarus and Verilator 5.054 on macOS: 420 pin/cycle records and
581742 cycles across six profiles against the C++ model, strict lint, the
pin-only cocotb test on six profiles, and the software suite. The SKY130A
qualification of v0.2.1 does not carry over to this RTL; synthesis, timing and
gate-level checks have not been rerun.

## v0.2.1 - 2026-11-05

First LPU4tt GitHub publication: English ISA/modeling documentation,
architecture diagram, compact reproduced experiments and current evidence index.
ISA v0.1 and qualified RTL are unchanged.

## Tiny-LPU v0.2.0

Curated compiler/model/RTL release with bounded SKY130A qualification for the
64B/one-lane user macro.
