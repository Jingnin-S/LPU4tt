# ISA convergence experiment contract (2026-11-03)

Independent device stream; ABI v1 stays supported. No RTL in this experiment.
Profiles: lanes 1/2/4 (default 2), depth 2..4 (default 3), scratch 32..256B,
memory_byte_bw default 2. A is broadcast, W has one read per lane per issue.
These are explicit port assumptions requiring later RTL/PPA validation.

## Semantics

Four opcodes: LOAD=1, STORE=2, MVM=3, END=15. Sequential completion, no instruction RAM,
DMA overlap, LOOP or descriptor table. Simulator commands and LOAD literals are separate
files; wire order is command followed by its LOAD literal, or STORE response. Host waits
for command completion before the next command. No file/config headers are charged as
wire traffic; configuration is explicit runtime/capability metadata.

LOAD/STORE addresses and lengths are bytes. Length field is count-1, range 1..256;
address+count must fit scratch. LOAD consumes the next literal bytes; STORE appends to
the response stream. Missing/extra LOAD payload, missing/duplicate END, trailing commands,
reserved bits, invalid opcodes and truncated encoding are errors.

MVM has byte A/W/Y addresses, M=field+1 (1..16), K=field+1 (1..64), and literal W row
stride in bytes, at least ceil(K*input_bits/8). Format 0=INT4, 1=INT8; INT8 requires an
explicit enable_int8 profile, all other formats rejected. A is packed contiguous; W rows
packed separately. INT4 low index is low nibble; odd-K high nibble ignored. All ranges
must fit. A/W read alias is legal; the union of Y read/write ranges must not alias A/W.
Y must be naturally aligned for its output and, if ACCUM, its partial-sum read.

Flags bit0 ACCUM: 0 clears initial accumulators; 1 reads signed little-endian accumulator-
width partial sums from Y. Bit1 SAT16: write clamp[-32768,32767] INT16, otherwise write raw
accumulator width. Format is flags[3:2]; flags[7:4] reserved zero. Every MAC explicitly
wraps signed two's complement on overflow and increments overflow_events. SAT16 counts
changed output values in saturation_events. ACC16 is a comparison profile; ACC32 is the
candidate. Compiler limits global K<=4096, for which INT4/INT8 zero-initialized sums fit
ACC32. This proof does not cover arbitrary unbounded ACCUM streams or arbitrary seeded Y.

Reset/new run clears state. Scratch starts uninitialized; model byte-valid checks reject
uninitialized reads (software validation, not a promise of per-byte RTL valid flops).
ACCUM reads the exact byte pattern; the compiler must not treat SAT16 output as an ACC32
partial sum. Errors are checked before the current command executes; preceding commands
may have had effects, no transactional rollback. Error returns no successful report.
External backpressure is at command/byte boundaries; an accepted MVM has fixed timing.
Electrical reset/handshake pin timing remains a later RTL contract.

## Actual encodings

Fixed64 (encoding=8): each command is 8 bytes `[op,flags,arg0,arg1,arg2,arg3,arg4,arg5]`.

- LOAD/STORE: `[op,0,addr,count_minus1,0,0,0,0]`.
- MVM: `[3,flags,a,w,y,M_minus1,K_minus1,stride_bytes]`.
- END: `[15,0,0,0,0,0,0,0]`.

Mixed32/64 (encoding=4): LOAD/STORE use `[op,0,addr,count_minus1]`, END uses
`[15,0,0,0]`, MVM stays 8 bytes. Opcode determines length; no varints, implicit descriptors
or free setup. Both encodings must roundtrip the same logical command stream exactly.

## Cycle contract

Each command decodes in one cycle. LOAD/STORE internal copy costs ceil(count/memory_byte_bw).
MVM processes groups of at most lanes rows: ACCUM first reads ceil(active*acc_bytes/bw),
then issues K products per lane over K ticks and drains depth-1 ticks, then writes
ceil(active*out_bytes/bw). Groups explicitly drain. Do not reuse the old overlapping-row
DseCore timing formula. Products must be formed from runtime scratch at issue and committed
through the delayed pipeline to lane accumulators, not computed by an oracle then backfilled.

Report core_cycles (issue+drain), scratch_cycles (partial reads/output writes/LOAD/STORE),
decode_cycles=command_count and device_cycles=sum. Conservative end-to-end seconds:
device_cycles/core_hz + (command_bytes+load_bytes+store_bytes)/link_Bps
+ host_ns_per_command*command_count/1e9. No overlap; 100/200MHz and 1/10/50MB/s are scenarios,
not GPIO/STA achievements. Return command_bytes from actual parsing.

## C ABI and verification

```c
const char* tlpu_isa_run(const unsigned char* commands, size_t command_bytes,
 const unsigned char* loads, size_t load_bytes, unsigned scratch_bytes,
 unsigned lanes, unsigned depth, unsigned accumulator_bits, unsigned memory_byte_bw,
 unsigned encoding, unsigned enable_int8);
const char* tlpu_isa_last_error(void);
```

Thread-local JSON valid until next run. On error return null plus last_error. Check all
parameters/pointers/lengths. JSON: stores_hex, command_count, mvm_count, macs, core_cycles,
scratch_cycles, decode_cycles, device_cycles, load_bytes, store_bytes, command_bytes,
overflow_events, saturation_events, peak_scratch_end. C++ owns decoding/ticked execution/
C ABI/unit tests. Python owns encoding/lowering/independent oracle/experiments/reports.

Cover M=1/3/7/8/16, odd tails, K=1/17/64/65/256/512/1024/4096, scratch=32/64/96/128/192/256B,
short-K W residency, long-K accumulation, ACC16 extreme counterexamples, optional INT8.
All compiler cycle/byte counts must match actual C++ execution; compare feasible short
cases with the old ABI too. Select ISA using coverage, measured stream overhead and
documented parser/architectural complexity; do not claim physical area or Fmax results.
