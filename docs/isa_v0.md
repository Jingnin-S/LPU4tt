# Tiny-LPU ISA v0.1

ISA v0.1 is an independent streaming integer instruction set for small
quantized linear layers and tiled GEMV. It is not Groq binary compatible.
The base arithmetic is signed INT4 x INT4 with INT32 accumulation and raw32
or explicitly saturating INT16 output. Repository version v0.2.1 does not
change the encoding. The complete model and RTL follow the
[device contract](device_rtl_contract.md).

## Capability profile

| Property | Base profile |
| --- | --- |
| Activation and weight | Signed INT4 in [-8, 7], zero-point 0 |
| Product and accumulator | Exact integer product; signed 32-bit accumulation |
| Output | Raw INT32 or explicit SAT16 INT16 |
| Addressing | 8-bit byte addresses; actual capacity is a target property |
| Per MVM | M = 1..16, K = 1..64, explicit W row stride |
| Compiler global shape | M = 1..64, K = 1..4096, batch = 1..32 through tiling |
| Execution | One in-order stream; LOAD literals follow their headers |
| Host responsibilities | Graphs, scales, quantization, nonlinear operations |

Capacity and lane count are not encoded as an implicit fixed microarchitecture.
The public wrapper uses 64B/one lane. 128B/two lanes is a model/RTL comparison,
and 192B is a longer-K modeling candidate. ACC16 and fixed 64-bit headers are
comparison profiles. INT8 requires explicit model capability; FP16/BF16 remain
numerical references.

## Instruction bytes

Bytes are listed in wire order, independently of host structure layout.
Opcode is the first byte and determines header length.

| Instruction | Bytes | Effect |
| --- | --- | --- |
| LOAD | 01 00 dst count-1 | Consume count following literal bytes into scratchpad |
| STORE | 02 00 src count-1 | Return count scratchpad bytes |
| MVM | 03 flags A W Y M-1 K-1 stride | Compute an addressed matrix-vector tile into Y |
| END | 0F 00 00 00 | Finish the stream and latch done |

LOAD/STORE count is 1..256. All accessed ranges must fit actual scratch capacity.
MVM stride is a literal byte count in 1..255, not a minus-one value, and must be
at least ceil(K * input_bits / 8). Reserved opcodes/bits are errors.
No further command is accepted after END until a new run.

| MVM flag bits | Meaning |
| --- | --- |
| 0 | ACCUM: 0 initializes from zero; 1 reads raw accumulator-width seeds at Y |
| 1 | SAT16: 0 writes raw accumulator width; 1 clamps final output to INT16 |
| 3:2 | 00 = INT4; 01 = optional INT8 model extension; 10/11 rejected |
| 7:4 | Reserved, must be zero |

The base device rejects INT8. Reserved formats are not a promise of floating
support. An added numeric profile must declare its version and capabilities.

## Data layout and arithmetic

- INT4 uses two's complement, lower-index element in the low nibble. A is packed
  contiguously; each W row starts at a byte boundary. Odd-K high nibbles are ignored
  and emitted as zero by the compiler. INT8 uses one signed byte per element.
- Y is little-endian. Raw32 requires 4B alignment; SAT16 requires 2B alignment;
  ACCUM also requires 4B alignment. A/W may alias as read-only operands.
  The union of Y seed/read and write ranges must not overlap A or the entire W
  address envelope, including row padding.
- With ACCUM=0 each output starts at zero. With ACCUM=1 the device loads the
  signed INT32 seed from Y + 4*row, then adds products in increasing K order.
  Seeds can come from a previous K tile or explicit host LOAD, including bias.
- Every MAC uses two's-complement wrap32 on overflow. Zero-initialized INT4/INT8
  dot products with total K <= 4096 fit INT32. That proof does not cover arbitrary
  host seeds or unbounded ACCUM sequences.
- SAT16 applies only at the end of the current MVM and clamps to [-32768,32767].
  It performs no scaling, rounding or ReLU. Intermediate K tiles must retain raw32.
  Compact SAT16 results cannot later be read as raw32 partial sums.
- Overflow/saturation event counters belong to verification models; the released
  hardware need not implement full event counters. Wrap/clamp results are mandatory.

## Ordering, initialization and errors

LOAD/STORE and MVM do not overlap. MVM executes at most one group of physical
lane-count rows at a time, draining and writing the group before advancing.
There is no hidden M-row accumulator bank.

Reset or an accepted start clears validity. All read bytes must have been
written in the current run. The complete device has per-byte validity; memory
data cells themselves are not reset. No free weight residency is assumed across
runs. END retains results until a new run invalidates scratchpad.

Illegal opcodes, reserved fields, truncation, out-of-range addresses, invalid
format/stride, uninitialized reads, Y alias/alignment errors, incomplete LOAD
payload and invalid END usage are rejected. Earlier commands may already have
effects; there is no transactional rollback. Device ERROR is sticky until
reset or disable. Start, reset, enable and handshake priorities are specified
in the [device contract](device_rtl_contract.md).

The software artifacts commands.bin and loads.bin store headers and literals
separately. The actual wire stream must interleave each LOAD header with its
literal bytes. Tensor IDs, scales, file names and manifests are host metadata,
not device instructions.

## K tiling example

A two-row dot product with global K=128 can use two K64 tiles, subject to
capacity and layout:

~~~text
LOAD W_block0
LOAD A_block0
MVM  flags=0x00, M=2, K=64    # zero initialize, raw32
LOAD W_block1
LOAD A_block1
MVM  flags=0x01, M=2, K=64    # ACCUM, raw32
STORE Y, 8 bytes
END
~~~

For final INT16, use flags=0x03 on the last MVM and STORE 4 bytes. Scratch
allocation still covers the full INT32 seed region. Row-group ordering makes
the final in-place narrowing safe.

## Timing models

The original ISA model uses one decode cycle, explicit byte-copy bandwidth,
K + depth - 1 compute cycles per group and serialized seed/writeback phases.
It is a controlled ISA trade-off model with idealized ports.

The implemented device separately models every header byte, boundary/check
stage, synchronous memory response, bank conflict, drain, writeback and handshake.
LOAD/STORE/END use header+2 control cycles; MVM uses header+3.
Use no_stall_cycles and C++ device ticks for implementation-aligned timing.
External stalls and host work must be added explicitly.

Implementation: compiler/tinylpu_compiler/isa.py and isa_graph.py,
hpp/tinylpu_isa.h, cpp/tinylpu_isa.cpp, and cpp/tinylpu_device.cpp.
See [experiments](isa_experiment_spec.md), [decisions](isa_results.md),
and [modeling](modeling.md).
