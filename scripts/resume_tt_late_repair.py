#!/usr/bin/env python3
"""MIT. Add official DRV repair after timing in an isolated checkpoint copy.

Every original Classic step and its gating remain in order. The extra step
uses LibreLane's unmodified RepairDesignPostGRT Tcl with separate margins.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def register_flow():
    from decimal import Decimal
    from librelane.config import Variable
    from librelane.flows import Flow
    from librelane.flows.classic import Classic
    from librelane.steps.openroad import RepairDesignPostGRT

    class LateRepair(RepairDesignPostGRT):
        id='OpenROAD.TinyLpuLateRepair'
        name='Electrical Repair after Timing'
        config_vars=RepairDesignPostGRT.config_vars+[
            Variable('TINYLPU_LATE_SLEW_MARGIN',Decimal,'Late DRV slew margin',default=Decimal(60)),
            Variable('TINYLPU_LATE_CAP_MARGIN',Decimal,'Late DRV cap margin',default=Decimal(50)),
            Variable('TINYLPU_ECO_DRIVER_PINS',list[str],'Finite post-RCX driver pin list',default=[]),
            Variable('TINYLPU_ECO_BUFFER_CELL',str,'Noninverting ECO buffer',default='sky130_fd_sc_hd__buf_8'),
            Variable('TINYLPU_ECO_MAX_LOADS',int,'Maximum loads per ECO buffer',default=8)]

        def get_script_path(self):
            if self.config['TINYLPU_ECO_DRIVER_PINS']:
                return '/workspace/project/targeted_eco.tcl'
            return super().get_script_path()

        def prepare_env(self,env,state):
            env=super().prepare_env(env,state)
            env['GRT_DESIGN_REPAIR_MAX_SLEW_PCT']=str(self.config['TINYLPU_LATE_SLEW_MARGIN'])
            env['GRT_DESIGN_REPAIR_MAX_CAP_PCT']=str(self.config['TINYLPU_LATE_CAP_MARGIN'])
            return env

    original=list(Classic.Steps)
    index=next(i for i,s in enumerate(original) if s.id=='OpenROAD.ResizerTimingPostGRT')+1

    @Flow.factory.register()
    class TinyLpuLateRepair(Classic):
        name='Tiny-LPU Classic with late electrical repair'
        Steps=original[:index]+[LateRepair]+original[index:]

    assert [s.id for s in TinyLpuLateRepair.Steps if s.id!=LateRepair.id]==[s.id for s in original]
    assert len(TinyLpuLateRepair.Steps)==len(original)+1
    return TinyLpuLateRepair


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--slew-margin',type=float,default=60)
    parser.add_argument('--cap-margin',type=float,default=50)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--eco-pins-file',type=Path)
    args=parser.parse_args()
    if args.inside:
        register_flow()
        sys.argv=['librelane','--flow','TinyLpuLateRepair','--manual-pdk','--pdk','sky130A',
                  '--pdk-root','/home/runner/pdk','--run-tag','wokwi','--force-run-dir','runs/wokwi',
                  '--from','OpenROAD.TinyLpuLateRepair','--with-initial-state',
                  '/workspace/project/initial_state.json','--jobs','4','--hide-progress-bar',
                  'src/config_merged.json']
        runpy.run_module('librelane',run_name='__main__')
        return
    if os.name!='posix':raise SystemExit('Run only on the Linux server')
    if not all(s.replace('_','').isalnum() for s in (args.base,args.name)):
        raise SystemExit('Simple candidate names required')
    if not all(0<=x<100 for x in (args.slew_margin,args.cap_margin)):
        raise SystemExit('Margins must be in [0,100) percent')
    pins=json.loads(args.eco_pins_file.read_text()) if args.eco_pins_file else []
    if pins and (not isinstance(pins,list) or len(pins)>256 or
                 any(not isinstance(p,str) or '/' not in p for p in pins)):
        raise SystemExit('Expected a finite JSON list of output pin names')
    base=ROOT/'physical/work'/args.base;work=ROOT/'physical/work'/args.name
    if work.exists():raise SystemExit('New candidate directory required; never overwrite')
    oldrun=base/'runs/wokwi'
    checkpoints=list(oldrun.glob('*-openroad-resizertimingpostgrt/state_out.json'))
    if len(checkpoints)!=1:raise SystemExit('Unique completed post-GRT timing checkpoint required')
    limit=int(checkpoints[0].parent.name.split('-')[0])
    work.mkdir(parents=True)
    shutil.copytree(base/'src',work/'src')
    for name in ('info.yaml','LICENSE'):shutil.copyfile(base/name,work/name)
    newrun=work/'runs/wokwi';newrun.mkdir(parents=True)
    for path in oldrun.iterdir():
        if path.is_dir() and path.name[:2].isdigit() and int(path.name.split('-')[0])<=limit:
            shutil.copytree(path,newrun/path.name)
        elif path.is_file() and path.suffix=='.json':shutil.copyfile(path,newrun/path.name)
        elif path.is_dir() and path.name=='tmp':shutil.copytree(path,newrun/path.name)
    shutil.copyfile(checkpoints[0],work/'initial_state.json')
    shutil.copyfile(Path(__file__),work/'flow_driver.py')
    if pins:shutil.copyfile(ROOT/'scripts/tt_targeted_eco.tcl',work/'targeted_eco.tcl')
    for name in ('config.json','config_merged.json'):
        path=work/'src'/name;config=json.loads(path.read_text())
        config['TINYLPU_LATE_SLEW_MARGIN']=args.slew_margin
        config['TINYLPU_LATE_CAP_MARGIN']=args.cap_margin
        config['MAX_THREADS']=4
        config['TINYLPU_ECO_DRIVER_PINS']=pins
        path.write_text(json.dumps(config,indent=2)+'\n')
    profile=json.loads((base/'candidate.json').read_text())
    profile.update(parent_candidate=args.base,checkpoint=str(checkpoints[0].relative_to(base)),
                   extra_flow_step='OpenROAD.TinyLpuLateRepair',
                   late_margins={'slew_pct':args.slew_margin,'cap_pct':args.cap_margin},
                   repair_mode='targeted_buffer' if pins else 'full_design',eco_driver_pins=pins,
                   signoff_passed=False)
    profile['implementation_sha256']={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (work/'src/config.json',work/'src/config_merged.json',
                                               work/'flow_driver.py',work/'initial_state.json')}
    if pins:profile['implementation_sha256']['targeted_eco.tcl']=hashlib.sha256((work/'targeted_eco.tcl').read_bytes()).hexdigest()
    (work/'candidate.json').write_text(json.dumps(profile,indent=2)+'\n')
    old_rtl=ROOT/'reports/tt_candidate_rtl'/args.base/'summary.json'
    receipt=json.loads(old_rtl.read_text())
    assert receipt['passed'] and all(h in receipt['source_sha256'].values() for h in profile['source_sha256'].values())
    receipt['inherited_from']=args.base
    new_rtl=ROOT/'reports/tt_candidate_rtl'/args.name;new_rtl.mkdir(parents=True)
    (new_rtl/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    subprocess.run(['git','init','-b','codex/tt-late-repair',str(work)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(work),'add','src','info.yaml','LICENSE','candidate.json','flow_driver.py'],check=True)
    if pins:subprocess.run(['git','-C',str(work),'add','targeted_eco.tcl'],check=True)
    subprocess.run(['git','-C',str(work),'-c','user.name=Tiny-LPU local flow',
                    '-c','user.email=tinylpu@localhost','commit','-m','Freeze late electrical repair candidate'],
                   check=True,stdout=subprocess.DEVNULL)
    command=['podman','--root','/tmp/tinylpu-tt-podman-overlay',
             '--runroot','/tmp/tinylpu-tt-podman-overlay-run','--storage-driver=overlay',
             '--storage-opt','overlay.ignore_chown_errors=true','run','--rm','--network=none',
             '--security-opt','label=disable','-v',f'{ROOT}:/work','-v',f'{work}:/workspace/project',
             '-v',f'{ROOT}/physical/vendor/tt-support-tools:/workspace/project/tt:ro',
             '-v',f'{ROOT}/physical/pdk:/home/runner/pdk:ro','-e','PDK_ROOT=/home/runner/pdk',
             '-w','/workspace/project','ghcr.io/librelane/librelane:3.0.14',
             'python3','/workspace/project/flow_driver.py','--inside','--base',args.base,'--name',args.name]
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_tt_candidate import watch
    with (work/'harden.log').open('w') as stream:code=watch(command,stream,work,12,3600)
    print(json.dumps({'candidate':args.name,'exit_code':code,'signoff_passed':False}),flush=True)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
