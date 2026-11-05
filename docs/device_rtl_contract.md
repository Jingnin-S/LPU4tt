# Complete device interface and cycle contract

The implementations are rtl/tinylpu_device_memory.sv, rtl/tinylpu_device.sv
and rtl/tt_um_tinylpu_v1.sv. cpp/tinylpu_device.cpp is the independent ticked
reference. Physical qualification is documented separately.

## Fixed functionality

ISA v0.1 uses 4B LOAD/STORE/END and 8B MVM headers. LOAD literals immediately
follow headers. STORE finishes its response before another header is accepted.
Arithmetic is W4A4/ACC32 with per-MAC wrap32 and final raw32/SAT16 output.
INT8/FP16/BF16 are not native device capabilities.

Parameterized RTL supports one/two lanes and capacities 32..256B divisible by
four. The released regression covers 64/128/256B x one/two lanes.
Four banks use address modulo four, one synchronous read each and one-cycle
read latency. There is one global byte write per clock. Per-byte valid bits
reject uninitialized reads; data cells are not reset.

Boundary validation uses full-width arithmetic, including 13-bit W end
calculations, before truncation to byte addresses. A/W read alias broadcasts;
Y may not overlap the A/W envelope. ACCUM seeds remain raw32.

## Tiny Tapeout pins

| Pin | Direction | Meaning |
| --- | --- | --- |
| ui_in[7:0] | Input | RX byte |
| uo_out[7:0] | Output | TX byte, zero when invalid |
| uio[0] | Input | RX valid |
| uio[1] | Input | TX ready |
| uio[2] | Output | RX ready |
| uio[3] | Output | TX valid |
| uio[4] | Output | Busy |
| uio[5] | Output | Done until reset/new start |
| uio[6] | Output | Sticky error until reset/disable |
| uio[7] | Input | Start, rising-edge event |

When enabled and out of reset, uio_oe is 0x7c. ena=0 or rst_n=0 immediately
gates outputs to zero and releases uio; the next clock edge clears state and
validity. State reset is synchronous and must span clock edges.

Transfer occurs when valid and ready are both high before a rising edge.
The source holds valid/data while stalled; TX obeys the same rule.
A new start edge has priority over same-edge RX/TX, so no transfer occurs on it.
IDLE/DONE accept start and invalidate scratch. Busy/ERROR start enters or
retains ERROR. Holding start high does not retrigger. DONE accepts no bytes.
Inputs must satisfy this clock's setup/hold; another clock domain needs an
external bridge.

## Exact no-stall cycle accounting

State order is HEADER -> ENDS -> MVM_VALIDATE (MVM only) -> CHECK, then dispatch.
LOAD/STORE/END cost header bytes + 2 control clocks; MVM costs header bytes + 3.
Incomplete headers or literals wait without a speculative timeout.

| Phase | Cost |
| --- | --- |
| LOAD literal | 1 clock/byte |
| STORE | 3 clocks/byte: request, synchronous response, TX handshake |
| MVM group initialization | 1 clock |
| ACCUM seed | 4 bytes/active lane, 2 clocks/byte |
| MVM core | Sum of K bank-request rounds + 3 drain clocks |
| Group writeback | 2 or 4 clocks/active lane |

At each k, coalesce equal A/W0/W1 addresses, count distinct addresses per bank,
and take the largest count as request rounds. Priority is A, W0, W1.
Two packet buffers overlap requests and prior responses; two product registers
feed ACC32. Conflict-free compute issues once per clock.

The interval from accepted start through DONE is one start clock plus all
instruction costs above. no_stall_cycles in compiler/tinylpu_compiler/device.py
is the independent phase sum. The 72-case performance experiment checks it
against actual C++ ticks. Random RX/TX stalls are additional.
Older ideal-port/resource cycles use different decode/copy assumptions.

## Functional and RTL evidence

run_device_vectors.py first checks compiler programs against independent ISA
arithmetic, then generates complete C++ pin traces. RTL checks all output pins
and direction. Independent arithmetic includes all 256 INT4 product pairs,
positive/negative accumulator wrap and saturation extremes.

The current source regression contains 402 records and 791566 cycles across
six profiles, with Icarus simulation and zero-warning strict Verilator lint.
Cases cover native K64, multi-group execution, K512/1024 tiling, odd K, tails,
packed/bank4 storage, conflicts, broadcasts, full-capacity transfer, invalid
headers/addresses/aliasing/uninitialized reads, reset/enable/start interruptions,
backpressure and restart after DONE.

The core SHA256 is
9fb9d085a29c4ee30b046583b4a6332206dbe13689f40644f4ae78160d5bd246.
Earlier 390/396-record results correspond to historical candidates.
The public pin-only cocotb test is independently arithmetic-driven.
Qualified routed gates additionally pass 67 records/137327 cycles and TT precheck.
See [reference evidence](../reports/README.md) and
[physical results](tt_signoff_results.md).

## Upstream execution boundary

DeviceRuntime drives the complete C++ byte interface from a compiled plan and
runtime input, then decodes STORE. It requires INT4/ACC32, one/two lanes,
depth three and one-byte interface timing; incompatible profiles are rejected.

Installed SGLang 0.5.22 ReplicatedLinear passed six shapes/twelve calls per
tested profile, including bias, skip-bias, post-load and an independent
quantized oracle. The public reference receipt records 64B/one-lane execution.
This is CPU software-to-model integration, not full SRT serving or silicon.
