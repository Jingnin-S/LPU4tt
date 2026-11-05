#!/usr/bin/env python3
"""Collect macro engineering gates; a completed LibreLane flow alone cannot pass."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--precheck',type=Path)
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--gate-vectors',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    candidate=(ROOT/args.candidate).resolve()
    final=candidate/'runs/wokwi/final'
    failures=[]
    def require(condition,reason):
        if not condition:failures.append(reason)
    require((final/'metrics.json').is_file(),'Final physical metrics missing')
    metrics=json.loads((final/'metrics.json').read_text()) if (final/'metrics.json').exists() else {}
    zeros=('design__inferred_latch__count','design__instance_unmapped__count',
           'route__drc_errors','magic__drc_error__count','design__lvs_error__count',
           'antenna__violating__nets','antenna__violating__pins',
           'magic__illegal_overlap__count','design__critical_disconnected_pin__count')
    checks={k:metrics.get(k) for k in zeros}
    for key,value in checks.items():require(value==0,f'{key}={value}')
    corners={}
    for rc in ('nom','min','max'):
        for pvt in ('tt_025C_1v80','ss_100C_1v60','ff_n40C_1v95'):
            corner=f'{rc}_{pvt}'
            values={key:metrics.get(f'{key}__corner:{corner}') for key in
                    ('timing__setup__ws','timing__hold__ws',
                     'design__max_slew_violation__count','design__max_cap_violation__count')}
            corners[corner]=values
            for key,value in values.items():
                valid=value is not None and math.isfinite(float(value))
                require(valid,f'{corner}: missing/invalid {key}')
                if valid:require(float(value)>=0 if key.startswith('timing__') else value==0,
                                 f'{corner}: {key}={value}')
    sources={p.name:digest(p) for p in (candidate/'src').glob('*.sv')}
    profile=json.loads((candidate/'candidate.json').read_text())
    require(sources==profile['source_sha256'],'Candidate RTL changed after freeze')
    wrapper=(candidate/'src/tt_um_tinylpu_v1.sv').read_text()
    for parameter,key in (('CAPACITY','capacity'),('LANES','lanes')):
        match=re.search(rf'\b{parameter}\s*=\s*(\d+)',wrapper)
        require(bool(match) and int(match[1])==profile.get(key),
                f'Actual synthesized {parameter} default differs from frozen profile')
    for relative,value in profile.get('implementation_sha256',{}).items():
        path=candidate/relative
        require(path.is_file() and digest(path)==value,f'Implementation changed after freeze: {relative}')
    config=json.loads((candidate/'src/config.json').read_text())
    period=float(config['CLOCK_PERIOD'])
    info=(candidate/'info.yaml').read_text()
    hz=int(re.search(r'clock_hz:\s*(\d+)',info)[1])
    require(abs(hz-1e9/period)<1,'Metadata clock differs from source constraint')
    sta_dirs=list((candidate/'runs/wokwi').glob('*-openroad-stapostpnr'))
    require(len(sta_dirs)==1,'Post-route STA directory missing/ambiguous')
    if len(sta_dirs)==1:
        reports=list(sta_dirs[0].glob('*/clock.rpt'))
        require(len(reports)==9,'Not all nine clock reports present')
        for report in reports:
            match=re.search(r'^Period:\s*([\d.]+)',report.read_text(),re.M)
            require(bool(match) and abs(float(match[1])-period)<1e-6,
                    f'Actual STA clock mismatch: {report.parent.name}')
    gds=final/'gds/tt_um_tinylpu_v1.gds'
    pnl=final/'pnl/tt_um_tinylpu_v1.pnl.v'
    lef=final/'lef/tt_um_tinylpu_v1.lef'
    artifacts={p.name:digest(p) for p in (gds,pnl,lef) if p.is_file()}
    require(len(artifacts)==3,'Final GDS/PNL/LEF missing')
    if args.gate:
        gate=json.loads((ROOT/args.gate).read_text())
        require(gate.get('passed') is True,'Gate simulation failed')
        require(gate.get('stage','routed')=='routed','Gate receipt is not from the routed netlist')
        require(pnl.is_file() and gate.get('netlist_sha256')==digest(pnl),'Gate netlist hash mismatch')
    else:require(False,'Gate simulation receipt missing')
    if args.gate_vectors:
        vectors=json.loads((ROOT/args.gate_vectors).read_text())
        require(vectors.get('passed') is True and vectors.get('all_numeric_checks_passed') is True,
                'Complete gate cycle vectors failed')
        require(vectors.get('capacity')==profile.get('capacity') and vectors.get('lanes')==profile.get('lanes'),
                'Gate vectors profile differs from physical candidate')
        require(vectors.get('cases',0)>0 and vectors.get('cycles',0)>0,'Empty gate cycle vectors')
        require(pnl.is_file() and vectors.get('netlist_sha256')==digest(pnl),'Gate vectors netlist hash mismatch')
    else:require(False,'Complete gate cycle vectors receipt missing')
    if args.precheck:
        precheck=json.loads((ROOT/args.precheck).read_text())
        require(precheck.get('passed') is True and precheck.get('checks')==15,'TT precheck not 15/15')
        bound=precheck.get('input_sha256',{})
        require(gds.is_file() and digest(gds) in bound.values(),'Precheck GDS hash unbound/mismatch')
        require(pnl.is_file() and digest(pnl) in bound.values(),'Precheck netlist hash unbound/mismatch')
        require(lef.is_file() and digest(lef) in bound.values(),'Precheck LEF hash unbound/mismatch')
    else:require(False,'Precheck receipt missing')
    rtl=ROOT/'reports/tt_candidate_rtl'/candidate.name/'summary.json'
    require(rtl.is_file(),'Exact candidate RTL simulation receipt missing')
    if rtl.exists():
        receipt=json.loads(rtl.read_text());hashes=receipt.get('source_sha256',{})
        require(receipt.get('passed') is True,'RTL test failed')
        require(receipt.get('capacity')==profile.get('capacity') and receipt.get('lanes')==profile.get('lanes'),
                'RTL test profile differs from frozen physical profile')
        for name,value in sources.items():
            require(value in hashes.values(),f'RTL test hash mismatch: {name}')
    report={'passed':not failures,'candidate':str(candidate.relative_to(ROOT)),
            'profile':profile,'period_ns':period,'checks':checks,'corners':corners,
            'frozen_candidate_sha256':digest(candidate/'candidate.json'),
            'frozen_profile_note':'The profile signoff_passed field is the pre-run placeholder; this report passed/failures is the qualification result.',
            'qualification_policy':'Finite nonnegative setup/hold and zero slew/cap in all nine corners; listed geometry/connectivity checks, exact RTL, routed pin-only and complete cycle-vector GL, and bound 15-check TT precheck. Other diagnostics are reported separately.',
            'diagnostics':{'max_fanout_violations_by_corner':{
                corner:metrics.get(f'design__max_fanout_violation__count__corner:{corner}') for corner in corners}},
            'receipt_sha256':{label:digest(ROOT/path) for label,path in
                              (('gate',args.gate),('gate_vectors',args.gate_vectors),('precheck',args.precheck)) if path},
            'area':{k:metrics.get(k) for k in ('design__core__area','design__die__area',
                                              'design__instance__area__stdcell',
                                              'design__instance__utilization__stdcell')},
            'source_sha256':sources,'artifacts_sha256':artifacts,'failures':failures,
            'scope':'User macro engineering checks, not chip integration, board rating or shuttle acceptance'}
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'candidate':report['candidate'],
                      'failed_gates':len(failures),'output':str(args.output)}))
    if args.strict and failures:raise SystemExit(1)


if __name__=='__main__':main()
