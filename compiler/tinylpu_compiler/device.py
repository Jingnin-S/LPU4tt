"""Cycle-model binding and wire-format helpers for the real TT byte protocol."""
import ctypes
import json
from pathlib import Path

from .isa import bind,unpack_stores

ENABLE=(1<<11)|(1<<12)


class Device:
    def __init__(self,library,capacity=128,lanes=2):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        self.lib.tlpu_device_create.argtypes=[ctypes.c_uint]*2
        self.lib.tlpu_device_create.restype=ctypes.c_void_p
        self.lib.tlpu_device_tick.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        self.lib.tlpu_device_tick.restype=ctypes.c_uint32
        self.lib.tlpu_device_peek.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        self.lib.tlpu_device_peek.restype=ctypes.c_uint32
        self.lib.tlpu_device_events.argtypes=[ctypes.c_void_p]
        self.lib.tlpu_device_events.restype=ctypes.c_uint32
        self.lib.tlpu_device_debug.argtypes=[ctypes.c_void_p]
        self.lib.tlpu_device_debug.restype=ctypes.c_char_p
        self.lib.tlpu_device_last_error.restype=ctypes.c_char_p
        self.lib.tlpu_device_destroy.argtypes=[ctypes.c_void_p]
        self.handle=self.lib.tlpu_device_create(capacity,lanes)
        if not self.handle:raise ValueError(self.lib.tlpu_device_last_error().decode())
    def tick(self,inputs):
        if not self.handle:raise RuntimeError('Closed device')
        output=self.lib.tlpu_device_tick(self.handle,inputs)
        if output==0xffffffff:raise RuntimeError(self.lib.tlpu_device_last_error().decode())
        return output,self.lib.tlpu_device_events(self.handle)
    def debug(self):return json.loads(self.lib.tlpu_device_debug(self.handle))
    def close(self):
        if self.handle:self.lib.tlpu_device_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def wire_bytes(plan,inputs):
    if plan['encoding']!=4:raise ValueError('Device base profile requires mixed32/64 encoding')
    commands,loads=bind(plan,inputs)
    result=bytearray();offset=cursor=0
    for command in plan['commands']:
        length=8 if command['op']=='MVM' else 4
        result.extend(commands[offset:offset+length]);offset+=length
        if command['op']=='LOAD':
            result.extend(loads[cursor:cursor+command['count']]);cursor+=command['count']
    assert offset==len(commands) and cursor==len(loads)
    return bytes(result)


def no_stall_cycles(plan,lanes):
    """Address-based cost of the frozen device FSM, including the accepted start.

    This is separate from either execution model. It assumes a synchronous host
    that holds every RX byte until ready and always accepts valid TX bytes.
    """
    if lanes not in (1,2) or plan['encoding']!=4 or plan['input_bits']!=4:
        raise ValueError('Cycle contract requires 1/2 lanes, mixed encoding, INT4')
    phases=dict(start=1,header=0,check=0,mvm_validate=0,load=0,store=0,group_init=0,
                seed=0,core_issue=0,core_drain=0,writeback=0)
    for command in plan['commands']:
        op=command['op']
        phases['header']+=8 if op=='MVM' else 4
        phases['check']+=2
        if op=='LOAD':phases['load']+=command['count']
        elif op=='STORE':phases['store']+=3*command['count']
        elif op=='MVM':
            phases['mvm_validate']+=1
            for row in range(0,command['rows'],lanes):
                active=min(lanes,command['rows']-row)
                phases['group_init']+=1
                if command['accumulate']:phases['seed']+=8*active
                phases['core_drain']+=3
                phases['writeback']+=active*(2 if command['sat16'] else 4)
                for k in range(command['k']):
                    addresses={command['a']+k//2}
                    addresses.update(command['w']+(row+i)*command['stride']+k//2
                                     for i in range(active))
                    requests=[sum(a%4==bank for a in addresses) for bank in range(4)]
                    phases['core_issue']+=max(requests)
        elif op!='END':raise ValueError('Unknown device opcode')
    return {'cycles':sum(phases.values()),'phases':phases,
            'assumption':'No external RX/TX stalls; includes start and complete device FSM'}


class DeviceRuntime:
    """ISA-runtime-shaped interface that actually drives the pin/cycle device."""
    def __init__(self,library):self.library=Path(library)

    def run(self,plan,inputs):
        p=plan['profile']
        if (plan['input_bits']!=4 or plan['encoding']!=4 or p['accumulator_bits']!=32 or
            p['lanes'] not in (1,2) or p['depth']!=3 or p['memory_byte_bw']!=1):
            raise ValueError('Device runtime requires INT4/ACC32, lanes1/2, depth3, byte-wide transport')
        cost=no_stall_cycles(plan,p['lanes']);wire=wire_bytes(plan,inputs)
        cursor=0;received=bytearray();cycles=1;done=False
        with Device(self.library,p['scratch_bytes'],p['lanes']) as device:
            device.tick(0);device.tick(ENABLE|(1<<10))
            for _ in range(cost['cycles']):
                inp=ENABLE|(1<<9)
                if cursor<len(wire):inp|=(1<<8)|wire[cursor]
                output,event=device.tick(inp);cycles+=1
                if event&1:cursor+=1
                if event&2:received.append((event>>8)&255)
                if output&(64<<8):raise ValueError(f'Device fault at input byte {cursor}: {device.debug()}')
                if output&(32<<8):done=True;break
            work=device.debug()
        assert done and cursor==len(wire) and cycles==cost['cycles'], 'Device cycle/stream mismatch'
        report={**work,'command_count':work['commands'],'device_cycles':cycles,
                'command_bytes':len(wire)-work['load_bytes'],'stores_hex':received.hex(),
                'cycle_breakdown':cost['phases'],'engine':'complete device pin/cycle C++ model'}
        return unpack_stores(plan,report),report
