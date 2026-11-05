#!/usr/bin/env python3
"""Run an isolated TT physical candidate using the already installed server flow.

Leaves every existing work directory and baseline result untouched. PDK/vendor
installation is separate; this script does not publish or submit anything.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import site
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def watch(command,stream,work,max_rss_gib,max_seconds):
    """Bound only this candidate; identify OpenROAD via UID and mounted inode."""
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
    started=time.monotonic();events=[]
    target=(work/'candidate.json').stat()
    while process.poll() is None:
        time.sleep(5)
        candidates=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid='],text=True).split()
        for pid_text in candidates:
            pid=int(pid_text);proc=Path('/proc')/pid_text
            try:
                status=(proc/'status').read_text()
                fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
                if not fields.get('Name','').startswith('.openroad'):continue
                if int(fields['Uid'].split()[0])!=os.getuid():continue
                mounted=(proc/'root/workspace/project/candidate.json').stat()
                if (mounted.st_dev,mounted.st_ino)!=(target.st_dev,target.st_ino):continue
                rss=int(fields.get('VmRSS','0').split()[0])
                if rss>max_rss_gib*1024*1024:
                    events.append({'pid':pid,'rss_kib':rss,'uid':os.getuid(),
                                   'candidate_inode':target.st_ino,'reason':'RSS limit'})
                    (work/'resource_guard.json').write_text(json.dumps(events,indent=2)+'\n')
                    os.kill(pid,signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if time.monotonic()-started>max_seconds:
            events.append({'reason':'Whole candidate wall time limit','seconds':max_seconds})
            (work/'resource_guard.json').write_text(json.dumps(events,indent=2)+'\n')
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill()
    return process.wait()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='lpu_fastarb_parent')
    parser.add_argument('--base',default='lpu_tt_um')
    parser.add_argument('--base-directory',type=Path,
                        help='Explicit TT project containing src/docs/test/info.yaml/LICENSE')
    parser.add_argument('--inside',choices=('rtl','config','harden'))
    parser.add_argument('--resume-config',action='store_true')
    parser.add_argument('--clock-period',type=float,default=10.0)
    parser.add_argument('--all-corners',action='store_true')
    parser.add_argument('--representative-corners',action='store_true')
    parser.add_argument('--ss-corner',action='store_true')
    parser.add_argument('--post-grt',action='store_true')
    parser.add_argument('--speed-cells',action='store_true')
    parser.add_argument('--strong-buffers',action='store_true')
    parser.add_argument('--grt-slew-margin',type=float)
    parser.add_argument('--grt-cap-margin',type=float)
    parser.add_argument('--grt-setup-margin',type=float)
    parser.add_argument('--clockgate-min-width',type=int,default=0)
    parser.add_argument('--tiles',choices=('1x1','1x2','2x2','3x2','4x2','6x2','8x2'),default='4x2')
    parser.add_argument('--max-rss-gib',type=float,default=12)
    parser.add_argument('--max-seconds',type=int,default=3600)
    parser.add_argument('--capacity',type=int,choices=(32,64,96,128,192,256),default=128)
    parser.add_argument('--lanes',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    if args.inside:
        if args.inside=='rtl':
            python=ROOT/'.venv-cocotb-container/bin/python'
            os.execv(str(python),[str(python),str(ROOT/'scripts/run_device_cocotb.py'),
                      '--source-dir',str(ROOT/'physical/work'/args.name/'src'),
                      '--capacity',str(args.capacity),'--lanes',str(args.lanes),
                      '--output',str(ROOT/'reports/tt_candidate_rtl'/args.name)])
        # PyPI cairocffi cannot discover Nix libraries via ldconfig. Load the
        # image's real Cairo library by its absolute path (no rendering stub).
        cairo=sorted(Path('/nix/store').glob('*-cairo-*/lib/libcairo.so.2'))
        if cairo:ctypes.CDLL(str(cairo[0]),mode=ctypes.RTLD_GLOBAL)
        site.addsitedir(str(ROOT/'physical/container-venv/lib/python3.13/site-packages'))
        os.environ['PATH']='/workspace/project/port-bin:'+os.environ['PATH']
        tool=Path('/workspace/project/tt/tt_tool.py')
        sys.path.insert(0,str(tool.parent))
        # Use the installed precheck-matching PDK, rather than letting LibreLane
        # auto-fetch its separate default PDK revision. This keeps all checks on.
        import tech
        tech.Sky130Tech.librelane_pdk_args='--manual-pdk --pdk sky130A'
        sys.argv=[str(tool),'--create-user-config' if args.inside=='config' else '--harden']
        if args.inside=='harden':sys.argv.append('--no-docker')
        runpy.run_path(str(tool),run_name='__main__')
        return
    if os.name!='posix':raise SystemExit('Run only on the Linux server')
    if not 2.0<=args.clock_period<=1000.0:raise SystemExit('Clock period out of range')
    if sum((args.all_corners,args.representative_corners,args.ss_corner))>1:
        raise SystemExit('Select one corner optimization policy')
    if args.max_rss_gib<=0 or args.max_seconds<=0:raise SystemExit('Positive resource bounds required')
    if args.clockgate_min_width<0:raise SystemExit('Clock-gating width must be nonnegative')
    if args.clockgate_min_width:
        raise SystemExit('Automatic clock gating is disabled: the pinned synthesis flow loses '
                         'scratch clear_valid priority. See physical/diagnostics/'
                         'cg8_clockgate_functional_diagnosis.txt. Existing failed candidates are preserved.')
    if any(x is not None and not 0<=x<100 for x in (args.grt_slew_margin,args.grt_cap_margin)):
        raise SystemExit('GRT DRV margins must be in [0,100) percent')
    if args.grt_setup_margin is not None and args.grt_setup_margin<0:
        raise SystemExit('GRT setup margin must be nonnegative')
    if not args.name.replace('_','').isalnum() or not args.base.replace('_','').isalnum():
        raise SystemExit('Work directory names must be simple identifiers')
    work=ROOT/'physical/work'/args.name
    base=args.base_directory.resolve() if args.base_directory else ROOT/'physical/work'/args.base
    sources=['tinylpu_device_memory.sv','tinylpu_device.sv','tt_um_tinylpu_v1.sv']
    if work.exists():
        if not args.resume_config or (work/'runs/wokwi').exists():
            raise SystemExit('Candidate exists; only a failed pre-hardening configuration can be resumed')
        for name in sources:
            if name=='tt_um_tinylpu_v1.sv':
                expected=(ROOT/'rtl'/name).read_text()
                expected=re.sub(r'CAPACITY=\d+',f'CAPACITY={args.capacity}',expected,count=1)
                expected=re.sub(r'LANES=\d+',f'LANES={args.lanes}',expected,count=1)
                assert (work/'src'/name).read_text()==expected,'Source changed'
            else:assert (work/'src'/name).read_bytes()==(ROOT/'rtl'/name).read_bytes(), 'Source changed'
    else:
        work.mkdir(parents=True)
        for name in ('src','docs','test'):
            shutil.copytree(base/name,work/name,ignore=shutil.ignore_patterns('sim_build','__pycache__','*.fst','*.vcd'))
        for name in ('info.yaml','LICENSE'):shutil.copyfile(base/name,work/name)
        for name in sources:shutil.copyfile(ROOT/'rtl'/name,work/'src'/name)
        wrapper=work/'src/tt_um_tinylpu_v1.sv'
        text=wrapper.read_text()
        text=re.sub(r'CAPACITY=\d+',f'CAPACITY={args.capacity}',text,count=1)
        text=re.sub(r'LANES=\d+',f'LANES={args.lanes}',text,count=1)
        wrapper.write_text(text)
    config_path=work/'src/config.json'
    config=json.loads(config_path.read_text())
    config['CLOCK_PERIOD']=args.clock_period
    config['MAX_THREADS']=4
    if args.all_corners:
        corners=[f'{rc}_{pvt}' for rc in ('nom','min','max')
                 for pvt in ('tt_025C_1v80','ss_100C_1v60','ff_n40C_1v95')]
        config['PNR_CORNERS']=corners
        config['RSZ_CORNERS']=corners
        config['MAX_TRANSITION_CONSTRAINT']=0.3
        config['PL_RESIZER_SETUP_SLACK_MARGIN']=1.0
    if args.representative_corners:
        corners=['nom_tt_025C_1v80','max_ss_100C_1v60','min_ff_n40C_1v95']
        config['PNR_CORNERS']=corners
        config['RSZ_CORNERS']=corners
        config['MAX_TRANSITION_CONSTRAINT']=0.5
        config['PL_RESIZER_SETUP_SLACK_MARGIN']=1.0
    if args.ss_corner:
        config['DEFAULT_CORNER']='max_ss_100C_1v60'
        config['PNR_CORNERS']=['max_ss_100C_1v60']
        config['RSZ_CORNERS']=[f'{rc}_{pvt}' for rc in ('nom','min','max')
                              for pvt in ('tt_025C_1v80','ss_100C_1v60','ff_n40C_1v95')]
        config['MAX_TRANSITION_CONSTRAINT']=0.75
        config['DESIGN_REPAIR_MAX_SLEW_PCT']=0
        config['DESIGN_REPAIR_MAX_CAP_PCT']=0
        config['PL_RESIZER_SETUP_SLACK_MARGIN']=1.0
    if args.post_grt:
        config['RUN_POST_GRT_DESIGN_REPAIR']=True
        config['RUN_POST_GRT_RESIZER_TIMING']=True
    if args.speed_cells:
        config['SYNTH_STRATEGY']='DELAY 2'
        config['SYNTH_SIZING']=True
        excluded=list(config.get('EXTRA_EXCLUDED_CELLS') or [])
        excluded.extend(['sky130_fd_sc_hd__clkdlybuf*','sky130_fd_sc_hd__dlymetal*'])
        config['EXTRA_EXCLUDED_CELLS']=excluded
    if args.clockgate_min_width:
        config['SYNTH_CLOCKGATE_MIN_WIDTH']=args.clockgate_min_width
        config['SYNTH_CLOCKGATE_POSEDGE_ICG']='sky130_fd_sc_hd__dlclkp_1/GATE/CLK/GCLK'
        config['SYNTH_CLOCKGATE_NEGEDGE_ICG']=None
    if args.strong_buffers:
        excluded=list(config.get('EXTRA_EXCLUDED_CELLS') or [])
        excluded.extend(f'sky130_fd_sc_hd__{cell}_{drive}'
                        for cell in ('buf','clkbuf') for drive in (1,2))
        config['EXTRA_EXCLUDED_CELLS']=excluded
    for key,value in (('GRT_DESIGN_REPAIR_MAX_SLEW_PCT',args.grt_slew_margin),
                      ('GRT_DESIGN_REPAIR_MAX_CAP_PCT',args.grt_cap_margin),
                      ('GRT_RESIZER_SETUP_SLACK_MARGIN',args.grt_setup_margin)):
        if value is not None:config[key]=value
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    info_path=work/'info.yaml'
    clock_hz=round(1e9/args.clock_period)
    info_text=re.sub(r'(clock_hz:\s*)\d+',lambda m:m[1]+str(clock_hz),info_path.read_text())
    info_text=re.sub(r'(tiles:\s*)"[0-9]+x[0-9]+"',lambda m:m[1]+json.dumps(args.tiles),info_text)
    description=f'{args.capacity}-byte, {args.lanes}-lane streaming INT4/ACC32 LPU physical-flow candidate.'
    info_text=re.sub(r'(description:\s*)"[^"\n]*"',lambda m:m[1]+json.dumps(description),info_text)
    info_path.write_text(info_text)
    hashes={}
    for name in sources:
        hashes[name]=hashlib.sha256((work/'src'/name).read_bytes()).hexdigest()
    (work/'candidate.json').write_text(json.dumps({'source_sha256':hashes,'base':args.base,
                                                  'base_directory':str(args.base_directory) if args.base_directory else None,
                                                  'clock_period_ns':args.clock_period,'clock_hz':clock_hz,
                                                  'all_corners_pnr':args.all_corners,
                                                  'representative_corners_pnr':args.representative_corners,
                                                  'ss_corner_pnr':args.ss_corner,
                                                  'post_grt_optimization':args.post_grt,
                                                  'speed_cells':args.speed_cells,
                                                  'strong_buffers':args.strong_buffers,
                                                  'grt_margins':{'slew_pct':args.grt_slew_margin,
                                                                 'cap_pct':args.grt_cap_margin,
                                                                 'setup_ns':args.grt_setup_margin},
                                                  'clockgate_min_width':args.clockgate_min_width,
                                                  'tiles':args.tiles,
                                                  'resource_limits':{'max_rss_gib':args.max_rss_gib,'max_seconds':args.max_seconds},
                                                  'capacity':args.capacity,'lanes':args.lanes,
                                                  'signoff_passed':False},indent=2)+'\n')
    (work/'port-bin').mkdir(exist_ok=True)
    port_tool=work/'port-bin/yowasp-yosys'
    if not port_tool.is_symlink():port_tool.symlink_to('/workspace/venv/bin/yowasp-yosys')
    if not (work/'.git').exists():
        subprocess.run(['git','init','-b','codex/tt-fastarb',str(work)],check=True)
        subprocess.run(['git','-C',str(work),'add','src','info.yaml','LICENSE','candidate.json'],check=True)
        subprocess.run(['git','-C',str(work),'-c','user.name=Tiny-LPU local flow',
                        '-c','user.email=tinylpu@localhost','commit','-m','Freeze isolated physical candidate'],check=True,
                       stdout=subprocess.DEVNULL)
    command=['podman','--root','/tmp/tinylpu-tt-podman-overlay',
             '--runroot','/tmp/tinylpu-tt-podman-overlay-run','--storage-driver=overlay',
             '--storage-opt','overlay.ignore_chown_errors=true','run','--rm','--network=none',
             '--security-opt','label=disable','-v',f'{ROOT}:/work',
             '-v',f'{work}:/workspace/project',
             '-v',f'{ROOT}/physical/vendor/tt-support-tools:/workspace/project/tt:ro',
             '-v',f'{ROOT}/physical/container-venv:/workspace/venv',
             '-v',f'{ROOT}/physical/pdk:/home/runner/pdk:ro',
             '-e','PDK=sky130A','-e','PDK_ROOT=/home/runner/pdk',
             '-e','OMP_NUM_THREADS=4','-w','/workspace/project',
             'ghcr.io/librelane/librelane:3.0.14','python3',
             '/work/scripts/run_tt_candidate.py']
    for action in ('rtl','config','harden'):
        with (work/f'{action}.log').open('w') as stream:
            invocation=command+['--inside',action,'--name',args.name,
                                '--capacity',str(args.capacity),'--lanes',str(args.lanes)]
            code=watch(invocation,stream,work,args.max_rss_gib,args.max_seconds) if action=='harden' else subprocess.run(invocation,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:raise SystemExit(f'{action} failed ({code}); see {work}/{action}.log')
        print(f'{args.name} {action} completed',flush=True)


if __name__=='__main__':main()
