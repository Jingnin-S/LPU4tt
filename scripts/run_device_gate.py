#!/usr/bin/env python3
"""Test the routed power-connected netlist with independent pin-only programs."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from cocotb_tools.runner import get_runner

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('routed','mapped'),default='routed')
    parser.add_argument('--unit-delay',type=int,choices=(0,1),default=1,
                        help='Uniform functional-cell delay in ns; not SDF or timing signoff')
    args=parser.parse_args()
    candidate=(ROOT/args.candidate).resolve()
    folder=(ROOT/args.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    if args.stage=='routed':
        netlist=candidate/'runs/wokwi/final/pnl/tt_um_tinylpu_v1.pnl.v'
    else:
        netlists=list((candidate/'runs/wokwi').glob('*-yosys-synthesis/tt_um_tinylpu_v1.nl.v'))
        assert len(netlists)==1,'Unique mapped synthesis netlist required'
        netlist=netlists[0]
    assert netlist.is_file()
    pdk=ROOT/'physical/pdk/sky130A/libs.ref/sky130_fd_sc_hd/verilog'
    wrapper=ROOT/'rtl/tb_tinylpu_gate.sv'
    runner=get_runner('icarus')
    defines={'FUNCTIONAL':1,'SIM':1}
    if args.stage=='routed':defines['USE_POWER_PINS']=1
    runner.build(sources=[pdk/'primitives.v',pdk/'sky130_fd_sc_hd.v',netlist,wrapper],
                 defines=defines,build_args=[f'-DUNIT_DELAY=#{args.unit_delay}'],
                 hdl_toplevel='tb_tinylpu_gate',build_dir=folder/'build',always=True,
                 timescale=('1ns','1ps'),log_file=folder/'compile.log')
    result=runner.test(test_module='test_device',hdl_toplevel='tb_tinylpu_gate',
                       test_dir=ROOT/'tests/rtl',build_dir=folder/'build',
                       results_xml=str(folder/'results.xml'),log_file=folder/'run.log')
    xml=ET.parse(result).getroot()
    failures=[x.attrib for x in xml.findall('.//failure')+xml.findall('.//error')]
    passed=bool(xml.findall('.//testcase')) and not failures
    summary={'passed':passed,'stage':args.stage,'unit_delay_ns':args.unit_delay,'netlist':str(netlist.relative_to(ROOT)),
             'netlist_sha256':hashlib.sha256(netlist.read_bytes()).hexdigest(),
             'test_sha256':hashlib.sha256((ROOT/'tests/rtl/test_device.py').read_bytes()).hexdigest(),
             'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),
             'test_cases':len(xml.findall('.//testcase')),'failures':failures,
             'scope':f'{args.stage} gate netlist functional simulation with UNIT_DELAY; STA/electrical results separate'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not passed:raise SystemExit(f'Gate regression failed; see {result}')


if __name__=='__main__':main()
