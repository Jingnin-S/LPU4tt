"""Pin-only cocotb tests usable unchanged with RTL or the TT gate netlist.

No shared-library or internal-state dependency: expected values use Python integers.
"""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly, Timer


class Host:
    def __init__(self, dut):
        self.dut = dut
        self.rng = random.Random(130)

    async def edge(self, data=0, valid=False, ready=False, start=False, rst=True, ena=True):
        d = self.dut
        await FallingEdge(d.clk)
        d.ui_in.value = data
        d.uio_in.value = int(valid) | (int(ready) << 1) | (int(start) << 7)
        d.rst_n.value = rst
        d.ena.value = ena
        await Timer(1, unit='ns')
        before, byte = int(d.uio_out.value), int(d.uo_out.value)
        await RisingEdge(d.clk)
        # Allow functional standard-cell UNIT_DELAY propagation before observing
        # status. Setup/hold/frequency are checked separately by routed STA.
        await Timer(10, unit='ns')
        await ReadOnly()
        return before, byte, int(d.uio_out.value)

    async def reset(self):
        for _ in range(3):
            await self.edge(rst=False)
            assert int(self.dut.uio_oe.value) == 0
        await self.edge()
        assert int(self.dut.uio_oe.value) == 0x7c

    async def run(self, wire, expected=b'', error=False, reset=True):
        if reset:
            await self.reset()
        await self.edge(start=True)
        cursor = 0
        received = bytearray()
        pending = False
        held_tx = None
        for _ in range(100000):
            if cursor < len(wire) and (pending or self.rng.random() > .3):
                pending = True
            ready = self.rng.random() > .5
            before, byte, after = await self.edge(wire[cursor] if pending else 0, pending, ready)
            assert int(self.dut.uio_oe.value) == 0x7c
            if held_tx is not None:
                assert before & 8 and byte == held_tx, 'TX changed while stalled'
            held_tx = byte if before & 8 and not ready else None
            if before & 4 and pending:
                cursor += 1
                pending = False
            if before & 8 and ready:
                received.append(byte)
            if after & 64:
                assert error, f'device error after {cursor} input bytes'
                return
            if after & 32:
                assert not error and cursor == len(wire)
                assert received == expected, (received.hex(), expected.hex())
                return
        raise AssertionError(f'Device timeout: wire={wire[:16].hex()}, rx={cursor}/{len(wire)}, '
                             f'tx={len(received)}, status=0x{after:02x}')


def load(address, payload):
    return bytes([1, 0, address, len(payload)-1]) + payload


def store(address, count):
    return bytes([2, 0, address, count-1])


END = bytes([15, 0, 0, 0])


@cocotb.test()
async def device_programs(dut):
    dut.ena.value = 0
    dut.rst_n.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    cocotb.start_soon(Clock(dut.clk, 100, unit='ns').start())
    host = Host(dut)
    # Read broadcast and same-bank contention, three rows including lane tail.
    wire = load(0, bytes([0x88]*24))
    wire += bytes([3, 0, 16, 0, 24, 2, 0, 8])  # K1, three row bytes at 0/8/16.
    wire += store(24, 12) + END
    await host.run(wire, (64).to_bytes(4, 'little')*3)
    # All signed nibble pairs are evaluated independently of model/RTL code.
    wire = bytearray()
    expected = bytearray()
    for a in range(-8, 8):
        for w in range(-8, 8):
            wire += load(0, bytes([a & 15])) + load(4, bytes([w & 15]))
            wire += bytes([3, 0, 0, 4, 8, 0, 0, 1]) + store(8, 4)
            expected += (a*w).to_bytes(4, 'little', signed=True)
    await host.run(wire+END, expected)
    # ACC32 wrap and final SAT16, with raw seed storage distinct from result width.
    for seed, a, sat in ((0x7fffffff, 1, False), (0x80000000, -1, False),
                          (32767, 1, True), (0xffff8000, -1, True)):
        result = (seed if seed < (1 << 31) else seed-(1 << 32))+a
        result = ((result+(1 << 31)) % (1 << 32))-(1 << 31)
        if sat:
            result = max(-32768, min(32767, result))
        size = 2 if sat else 4
        wire = load(0, bytes([a & 15])) + load(4, bytes([1])) + load(8, seed.to_bytes(4, 'little'))
        wire += bytes([3, 1 | (int(sat) << 1), 0, 4, 8, 0, 0, 1]) + store(8, size) + END
        await host.run(wire, result.to_bytes(size, 'little', signed=True))
    await host.run(END, reset=False)
    # Odd K: the unused high nibbles of the last A/W bytes must not contribute.
    wire = load(0, bytes([0x21, 0x73])) + load(4, bytes([0x11, 0x71]))
    wire += bytes([3, 0, 0, 4, 8, 0, 2, 2]) + store(8, 4) + END
    await host.run(wire, (6).to_bytes(4, 'little'))
    # Two rows through seeds, SAT16 writeback and a multi-byte STORE.
    wire = load(0, bytes([0x77]*2)) + load(4, bytes([0x77]*2)) + load(8, bytes([0x88]*2))
    wire += load(16, (32700).to_bytes(4, 'little') + (-32700).to_bytes(4, 'little', signed=True))
    wire += bytes([3, 3, 0, 4, 16, 1, 3, 4]) + store(16, 4) + END
    await host.run(wire, (32767).to_bytes(2, 'little') + (-32768).to_bytes(2, 'little', signed=True))
    # A new start invalidates old scratch; invalid headers and uninitialized reads fault.
    # The last case faults on the second, uninitialized byte of a STORE.
    for wire in (store(0, 1), bytes([3, 0, 0, 255, 32, 15, 63, 255]), bytes([99, 0, 0, 0]),
                 load(0, bytes([1])) + store(0, 2)):
        await host.run(wire, error=True)
    # Busy start abort and ena recovery, tested through the external pins only.
    await host.reset()
    await host.edge(start=True)
    await host.edge(data=3, valid=True)
    _, _, after = await host.edge(start=True, valid=True, ready=True)
    assert after & 64
    await host.edge(ena=False)
    assert int(dut.uio_out.value) == 0 and int(dut.uio_oe.value) == 0
    await host.edge()
    await host.run(END, reset=False)
