#!/usr/bin/env python3
"""Measure final-device cycles for compiler-selected plans; do not predict Fmax."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compiler'))
from tinylpu_compiler.isa import IsaRuntime,Profile,bind
from tinylpu_compiler.architecture import ArchRuntime,Resources,choose_plan
from tinylpu_compiler.device import Device,ENABLE,wire_bytes,no_stall_cycles


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/device_performance')
    args=parser.parse_args()
    folder=(ROOT/args.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    isa=IsaRuntime(ROOT/'build_esl/libtinylpu_isa.so')
    analyzer=ArchRuntime(ROOT/'build_esl/libtinylpu_architecture.so')
    rows=[]
    for m,k,frames in ((8,16,1),(8,16,8),(8,64,1),(8,64,8),(3,65,2),(3,256,1)):
        # Identical weights/inputs for every configuration and storage candidate.
        rng=random.Random(10000*m+100*k+frames)
        weights=[[rng.randrange(-8,8) for _ in range(k)] for _ in range(m)]
        inputs=[[rng.randrange(-8,8) for _ in range(k)] for _ in range(frames)]
        for capacity in (64,128,256):
            for lanes in (1,2):
                p=Profile(capacity,lanes,3,32,1)
                for storage in ('packed','bank4'):
                    plan,estimate,candidates=choose_plan(weights,frames,p,Resources(),analyzer,storage,16)
                    commands,loads=bind(plan,inputs)
                    expected=bytes.fromhex(isa.run_raw(commands,loads,p,4)['stores_hex'])
                    wire=wire_bytes(plan,inputs);cursor=0;received=bytearray()
                    with Device(ROOT/'build_esl/libtinylpu_device.so',capacity,lanes) as device:
                        device.tick(0);device.tick(ENABLE|(1<<10));cycles=1
                        for _ in range(1000000):
                            inp=ENABLE|(1<<9)
                            if cursor<len(wire):inp|=(1<<8)|wire[cursor]
                            output,event=device.tick(inp);cycles+=1
                            if event&1:cursor+=1
                            if event&2:received.append((event>>8)&255)
                            assert not output&(64<<8)
                            if output&(32<<8):break
                        else:raise AssertionError('Device timeout')
                        assert cursor==len(wire) and received==expected
                        work=device.debug()
                    cost=no_stall_cycles(plan,lanes)
                    assert cycles==cost['cycles'],(m,k,frames,capacity,lanes,storage,cycles,cost)
                    record=dict(m=m,k=k,frames=frames,capacity=capacity,lanes=lanes,storage=storage,
                                tile_rows=plan['tile_rows'],tile_k=plan['tile_k'],order=plan['order'],
                                peak_bytes=plan['layout']['end'],candidate_count=len(candidates),
                                device_cycles=cycles,macs=work['macs'],wire_input_bytes=len(wire),
                                output_bytes=len(received),lane_utilization=work['macs']/(lanes*cycles),
                                **{f'cycles_{key}':v for key,v in cost['phases'].items()})
                    rows.append(record)
    with (folder/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(passed=True,configurations=len(rows),
                 planner='Existing resource+link objective selects each executable candidate; measured device cycles are separate',
                 assumptions='No external stalls; all compiler/ISA integer results and analytical full-device cycle counts checked',
                 physical_fmax_validated=False,
                 source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                for p in ('cpp/tinylpu_device.cpp','compiler/tinylpu_compiler/device.py',
                                          'scripts/run_device_performance.py')},
                 note='Compare cycles within identical m/k/frames. Runtime wall time is simulation, not hardware latency.')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
