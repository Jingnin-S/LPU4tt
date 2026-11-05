# Architecture experiment contract — 2026-11-04

Preserve ISA v0.1 and old reports. This stage adds explicit resource timing,
new ISA Relax VM backend, compiler composition/debugging and RTL *probes*.
All builds/tests/experiments execute on the USC server. Local files are mirrors.

## Resource model, C++ first

New independent library `tinylpu_architecture` and CTest. C ABI:
```
const char* tlpu_arch_analyze(const unsigned char* commands, size_t command_bytes,
 unsigned encoding, unsigned lanes, unsigned depth, unsigned acc_bits,
 unsigned banks, unsigned flat_read_ports, unsigned read_latency,
 unsigned copy_bytes_per_cycle);
const char* tlpu_arch_last_error(void);
```
There are eight unsigned configuration arguments after the byte count.
Thread-local JSON/error, exceptions contained. This is a resource analyzer, not
a replacement for ISA numerical execution/initialization checks.

Validate mixed/fixed stream lengths, reserved bits, END, M/K/stride/format, range
within 256B, Y alignment/nonalias, config lanes1/2/4/depth2..4/acc16/32,
banks in {0,1,2,4}, flat ports1..5, latency0..2, copy BW1..4.
banks=0 means flat shared array with `flat_read_ports`; otherwise modulo-byte
interleaving `bank=address % banks`, one read port per bank.
For each row group and k index gather unique BYTE addresses of A plus active W
(INT4 k//2, INT8 k). Equal addresses broadcast without duplicate port usage.
Rounds = ceil(unique/ports) flat, or max(count per bank) banked. Next index starts
after these rounds. Memory responses can pipeline: read_latency adds once per
group, not once per element. Each response set issues one MAC per active lane;
drain depth-1 after the last issue. Thus group cycles=sum(rounds)+read_latency+
depth-1. No prefetch across row groups. Count operand_read_bytes, gather_cycles,
conflict_cycles=sum(rounds-1), memory_latency_cycles, pipeline_drain_cycles.
Partial-sum reads/output writes are serialized phases (BW config), as are
LOAD/STORE internal copies. decode1 per instruction. JSON fields:
command_count, command_bytes, mvm_count, macs, load_bytes, store_bytes,
operand_read_bytes, gather_cycles, conflict_cycles, memory_latency_cycles,
pipeline_drain_cycles, core_cycles, scratch_cycles, decode_cycles, device_cycles.
core_cycles=gather+latency+drain; device=core+scratch+decode. Unique byte reads
are counted per k index even if consecutive INT4 indices use the same byte;
no undocumented operand cache or nibble-pair reuse.

Tests: flat ports1/2/3, bank conflicts and equal-address broadcast, stride/padding,
K tails/row tails, latency, partial sums/SAT16, exact counts, illegal streams.
Only after C++ validation may the matching memory/MAC RTL probes be synthesized.

## Compiler storage policy

Compare `packed` legacy row stride with `bank4` layout: W base=0, row stride is
smallest >=packed row bytes with stride %4=2; reserve padded lane rows times
stride. A base is smallest >=W reservation with A%4=1; Y follows A, naturally
aligned to 4 for partial/raw32, otherwise2. With lanes2, INT4 or INT8,
W group banks differ by2 and A differs by odd offset for every K index, so
all three reads fit distinct banks. This proof applies to compiler layout,
not arbitrary ISA programs. Padding consumes real storage and LOAD payload.
Other lane counts use resource model conflicts (no promise of II=1).

Lowering first generates the existing real ISA schedule, then relocates disjoint
regions and repacks each W LOAD including stride padding. Update count/offset/
bindings, actual bytes, hashes, prediction and store indices. Search only legal
layouts; numerical C++ runtime verifies every selected plan.

## Composition / upstream adaptation

Real torch.export -> Relax partition -> DAG outputs -> new ISA packed callbacks
executed by actual Relax VM. Support existing linear and explicit host ReLU/Q34
chains, plus tuple Q/K/V siblings. Default old frontend behavior stays unchanged.
Compare separate kernels and concatenated-weight QKV fusion; output slicing stays
host/Relax. Common source is required for fusion. Requant is a host dependency;
never fuse across it. Batch1/8 represents decode-like/prefill-like shapes, not a
complete attention implementation. Compiler searches lane/capacity/layout/order/
tile shape, with resource and byte costs visible. Trace per logical node:
placement, partition mapping, A/W/Y, command indices, dtype/host boundary, actual
statistics, resource timing, output hash; disassemble actual binary. Negative
tests reject incompatible fusion, unsafe INT16 range and unsupported operators.
SGLang adaptation may expose a CPU model/linear bridge for offline testing;
do not claim a running SGLang scheduler, GPU backend or production LLM accuracy.

## RTL/PPA probes

Server tool inventory first. Synthesizable Verilog probes for flat 1/2/3-read
byte array and 2/4-bank arrays, single byte write, capacities32/64/96/128/192/256,
and signed INT4 MAC with ACC16/32, lanes1/2/4, registered read/multiply/accumulate.
Test probes against C++/independent vectors before synthesis. Generic Yosys cells
and mapped liberty area are different metrics. No Fmax from cell counts, ABC -D
or unmapped logic. If usable liberty/STA exist, report exact library/corner,
constraints, no-wire vs routed assumptions; do not install giant PDK silently.
If no characterized flow exists, give resource/area evidence and explicit
physical signoff conditions; never invent a clock result.

## Proposed complete transport contract

One clock domain; ui_in RX byte, uo_out TX byte. uio[0]=rx_valid input,
[1]=tx_ready input,[2]=rx_ready output,[3]=tx_valid output,[4]=busy output,
[5]=done output,[6]=error output,[7]=start input; selected uio_oe=0x7c.
Transfer at rising edge valid&&ready, data stable under stall. start accepted only
IDLE/DONE; start while busy enters ERROR. END sets DONE until next start/reset.
rst_n synchronous active-low; host holds reset >=2 rising edges. ena=0 aborts,
clears validity/control and tri-states uio, re-enable requires start. No clock
gating. Host synchronous to clk; a different clock requires a future CDC bridge.
Header scratch=8B, no instruction RAM/FIFO. LOAD bytes write scratch on handshake,
STORE holds output stable until accepted. Invalid header enters sticky ERROR;
no rollback. A partial header/payload can wait indefinitely; host watchdog resets.
Out-of-band sideband status needs no ACK byte; turnaround gaps explicit in model.
The resource report remains conservative additive (external link plus internal
copies); a separate streaming bound may remove LOAD/STORE duplicate copy cycles,
never mix both definitions without labels.
