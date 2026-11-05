#!/usr/bin/env python3
"""Run unmodified pinned TT precheck in an isolated report directory on server."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import site
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gds',required=True,type=Path)
    parser.add_argument('--label',required=True)
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    if not args.label.replace('_','').isalnum():raise SystemExit('Use a simple report label')
    folder=ROOT/'physical/work/parent_checks'/args.label
    if args.inside:
        # Load real Nix C++ libraries for the PyPI layout bindings.
        for pattern in ('*-gcc-*-lib/lib/libgcc_s.so.1','*-gcc-*-lib/lib/libstdc++.so.6'):
            libraries=sorted(Path('/nix/store').glob(pattern))
            if libraries:ctypes.CDLL(str(libraries[0]),mode=ctypes.RTLD_GLOBAL)
        site.addsitedir(str(ROOT/'physical/container-venv/lib/python3.13/site-packages'))
        sys.path.insert(0,str(folder/'precheck'))
        os.environ['PATH']=str(folder/'port-bin')+':'+os.environ['PATH']
        sys.argv=[str(folder/'precheck/precheck.py'),'--gds',str(args.gds),'--tech','sky130A']
        runpy.run_path(str(folder/'precheck/precheck.py'),run_name='__main__')
        return
    if os.name!='posix':raise SystemExit('Run on Linux server')
    gds=args.gds.resolve()
    if ROOT not in gds.parents or not gds.is_file():raise SystemExit('GDS must be in the project')
    inputs=[gds,gds.with_suffix('.lef'),gds.with_suffix('.v')]
    input_sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if folder.exists():raise SystemExit('Use a new report label; never overwrite prior checks')
    folder.mkdir(parents=True)
    shutil.copytree(ROOT/'physical/vendor/tt-support-tools/precheck',folder/'precheck',
                    ignore=shutil.ignore_patterns('reports','__pycache__'))
    (folder/'precheck/reports').mkdir()
    (folder/'tech').symlink_to(os.path.relpath(ROOT/'physical/vendor/tt-support-tools/tech',folder))
    (folder/'port-bin').mkdir()
    (folder/'port-bin/yowasp-yosys').symlink_to('/workspace/venv/bin/yowasp-yosys')
    command=['podman','--root','/tmp/tinylpu-tt-podman-overlay',
             '--runroot','/tmp/tinylpu-tt-podman-overlay-run','--storage-driver=overlay',
             '--storage-opt','overlay.ignore_chown_errors=true','run','--rm','--network=none',
             '--security-opt','label=disable','-v',f'{ROOT}:/work',
             '-v',f'{ROOT}/physical/container-venv:/workspace/venv',
             '-v',f'{ROOT}/physical/pdk:/home/runner/pdk:ro',
             '-e','PDK=sky130A','-e','PDK_ROOT=/home/runner/pdk','-e','QT_QPA_PLATFORM=offscreen',
             '-w',f'/work/{folder.relative_to(ROOT)}/precheck',
             'ghcr.io/librelane/librelane:3.0.14','python3','/work/scripts/run_tt_precheck.py',
             '--inside','--gds',f'/work/{gds.relative_to(ROOT)}','--label',args.label]
    with (folder/'precheck.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(f'Precheck failed: see {folder}/precheck.log')
    xml=ET.parse(folder/'precheck/reports/results.xml').getroot()
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    cases=len(xml.findall('.//testcase'));assert cases>=15
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in input_sha.items()), 'Inputs changed during precheck'
    summary={'passed':True,'checks':cases,'gds':str(gds.relative_to(ROOT)),
             'input_sha256':input_sha,
             'scope':'Official Tiny Tapeout precheck; all applicable SKY130A checks retained'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
