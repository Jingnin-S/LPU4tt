#!/usr/bin/env python3
"""Reproduce independent integer checks and full pin/cycle co-simulation on server."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)} failed ({result.returncode}); see {log}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', default='64:1,64:2,128:1,128:2,256:1,256:2')
    parser.add_argument('--output', default='reports/device_regression')
    parser.add_argument('--vcs-module', default='vcs/X-2025.06')
    parser.add_argument('--simulator', choices=('vcs', 'iverilog'), default='vcs')
    parser.add_argument('--vectors-root', type=Path,
                        help='Reuse verified host Python3.11 vectors in a simulator-only container')
    args = parser.parse_args()
    folder = (ROOT / args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for profile in args.profiles.split(','):
        capacity, lanes = map(int, profile.split(':'))
        case = folder / f'c{capacity}_l{lanes}'
        case.mkdir(exist_ok=True)
        if args.vectors_root:
            vector_source = args.vectors_root.resolve()/case.name
            for name in ('summary.json', 'cycles.txt'):
                shutil.copyfile(vector_source/name, case/name)
            model_summary = json.loads((case/'summary.json').read_text())
            assert (model_summary['capacity'], model_summary['lanes']) == (capacity, lanes)
            assert hashlib.sha256((case/'cycles.txt').read_bytes()).hexdigest() == model_summary['vectors_sha256']
        else:
            run([sys.executable, str(ROOT/'scripts/run_device_vectors.py'), '--capacity', str(capacity),
                 '--lanes', str(lanes), '--output', str(case)], case/'model.log')
        sources = ['rtl/tinylpu_device_memory.sv', 'rtl/tinylpu_device.sv',
                   'rtl/tt_um_tinylpu_v1.sv', 'rtl/tb_tinylpu_device.sv']
        if args.simulator == 'vcs':
            command = ['vcs', '-full64', '-sverilog', '-timescale=1ns/1ps',
                       '-top', 'tb_tinylpu_device', f'-pvalue+tb_tinylpu_device.CAPACITY={capacity}',
                       f'-pvalue+tb_tinylpu_device.LANES={lanes}', *sources,
                       '-o', str(case/'simv'), f'-Mdir={case}/csrc']
            run(['bash', '-lc', f'module load {shlex.quote(args.vcs_module)} && {shlex.join(command)}'],
                case/'compile.log')
            simulation = [str(case/'simv'), '-no_save', f'+VECTORS={case}/cycles.txt']
        else:
            run(['iverilog', '-g2012', '-s', 'tb_tinylpu_device',
                 f'-Ptb_tinylpu_device.CAPACITY={capacity}', f'-Ptb_tinylpu_device.LANES={lanes}',
                 '-o', str(case/'simv.vvp'), *sources], case/'compile.log')
            simulation = ['vvp', str(case/'simv.vvp'), f'+VECTORS={case}/cycles.txt']
            run(['verilator', '--lint-only', '--top-module', 'tt_um_tinylpu_v1', '-Wall',
                 f'-GCAPACITY={capacity}', f'-GLANES={lanes}', *sources[:3]], case/'lint.log')
        output = run(simulation, case/'rtl.log')
        summary = json.loads((case/'summary.json').read_text())
        match = re.search(r'DEVICE RTL PASS cycles=(\d+)', output)
        assert match and int(match[1]) == summary['cycles'], case
        record = {k: summary[k] for k in ('capacity', 'lanes', 'cases', 'cycles', 'vectors_sha256')}
        record['passed'] = True
        records.append(record)
        print(json.dumps(record), flush=True)
    sources = ['rtl/tinylpu_device_memory.sv', 'rtl/tinylpu_device.sv',
               'rtl/tt_um_tinylpu_v1.sv', 'rtl/tb_tinylpu_device.sv', 'cpp/tinylpu_device.cpp',
               'hpp/tinylpu_device.h', 'scripts/run_device_vectors.py']
    report = {'passed': True, 'simulator': args.simulator, 'profiles': records,
              'total_cycles': sum(x['cycles'] for x in records),
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'scope': 'RTL functional and cycle checks, without physical timing or signoff claims'}
    (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
