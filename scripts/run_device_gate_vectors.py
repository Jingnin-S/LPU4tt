#!/usr/bin/env python3
"""MIT. Run every verified device cycle vector on the routed powered netlist."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--vectors-root',type=Path,default=ROOT/'reports/device_mvmvalidate_vectors')
    parser.add_argument('--model-report',type=Path,default=ROOT/'reports/device_mvmvalidate/summary.json')
    args=parser.parse_args()
    candidate=(ROOT/args.candidate).resolve()
    profile=json.loads((candidate/'candidate.json').read_text())
    model=json.loads(args.model_report.read_text());assert model['passed']
    for name in ('cpp/tinylpu_device.cpp','hpp/tinylpu_device.h','scripts/run_device_vectors.py'):
        assert digest(ROOT/name)==model['source_sha256'][name],f'Model source changed: {name}'
    for name in ('tinylpu_device.sv','tinylpu_device_memory.sv'):
        assert digest(candidate/'src'/name)==model['source_sha256'][f'rtl/{name}']
    cap,lanes=profile['capacity'],profile['lanes']
    vectors=args.vectors_root/f'c{cap}_l{lanes}'
    reference=json.loads((vectors/'summary.json').read_text())
    assert reference['all_numeric_checks_passed']
    assert digest(vectors/'cycles.txt')==reference['vectors_sha256']
    record=next(x for x in model['profiles'] if x['capacity']==cap and x['lanes']==lanes)
    assert record['passed'] and record['vectors_sha256']==reference['vectors_sha256']
    output=(ROOT/args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    pnl=candidate/'runs/wokwi/final/pnl/tt_um_tinylpu_v1.pnl.v'
    pdk=ROOT/'physical/pdk/sky130A/libs.ref/sky130_fd_sc_hd/verilog'
    bench=ROOT/'rtl/tb_tinylpu_gate_vectors.sv'
    sources=[pdk/'primitives.v',pdk/'sky130_fd_sc_hd.v',pnl,bench]
    command=['iverilog','-g2012','-s','tb_tinylpu_gate_vectors','-DFUNCTIONAL','-DUSE_POWER_PINS',
             '-DSIM','-DUNIT_DELAY=#1','-o',str(output/'gate.vvp'),*map(str,sources)]
    with (output/'compile.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    run=subprocess.run(['vvp',str(output/'gate.vvp'),f'+VECTORS={vectors}/cycles.txt'],
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'run.log').write_text(run.stdout)
    match=re.search(r'DEVICE GATE VECTORS PASS cycles=(\d+)',run.stdout)
    passed=run.returncode==0 and bool(match) and int(match[1])==reference['cycles']
    summary={'passed':passed,'capacity':cap,'lanes':lanes,'cases':reference['cases'],
             'cycles':reference['cycles'],'all_numeric_checks_passed':True,
             'netlist_sha256':digest(pnl),'bench_sha256':digest(bench),
             'vectors_sha256':reference['vectors_sha256'],'model_report_sha256':digest(args.model_report),
             'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
             'scope':'Full routed functional/cycle vectors with UNIT_DELAY=1, clock100ns/observe20ns; physical timing is checked separately by extracted STA'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not passed:raise SystemExit(f'Gate vectors failed; see {output}/run.log')


if __name__=='__main__':main()
