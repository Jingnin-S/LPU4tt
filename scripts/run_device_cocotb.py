#!/usr/bin/env python3
"""Run pin-only independent arithmetic tests; requires cocotb2 and Icarus."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cocotb
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=128)
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--output', default='reports/device_cocotb')
    parser.add_argument('--source-dir',type=Path,default=ROOT/'rtl')
    args = parser.parse_args()
    folder = (ROOT/args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source_dir=args.source_dir.resolve()
    paths=[source_dir/name for name in ('tinylpu_device_memory.sv','tinylpu_device.sv','tt_um_tinylpu_v1.sv')]
    runner = get_runner('icarus')
    runner.build(sources=paths, hdl_toplevel='tt_um_tinylpu_v1',
                 parameters={'CAPACITY': args.capacity, 'LANES': args.lanes},
                 build_dir=folder/'build', always=True, timescale=('1ns', '1ps'),
                 log_file=folder/'compile.log')
    result = runner.test(test_module='test_device', hdl_toplevel='tt_um_tinylpu_v1',
                         test_dir=ROOT/'tests/rtl', build_dir=folder/'build',
                         results_xml=str(folder/'results.xml'), log_file=folder/'run.log')
    xml = ET.parse(result).getroot()
    assert len(xml.findall('.//testcase')) > 0, 'No cocotb tests discovered'
    assert not xml.findall('.//failure') and not xml.findall('.//error'), f'See {result}'
    summary = {'passed': True, 'capacity': args.capacity, 'lanes': args.lanes,
               'simulator': 'iverilog', 'cocotb': cocotb.__version__,
               'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths+[ROOT/'tests/rtl/test_device.py']},
               'scope': 'Pin-only RTL test, independent arithmetic expectations; no STA claim'}
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
