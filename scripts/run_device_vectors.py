#!/usr/bin/env python3
"""Independently validate full-device C++ numerics, generate pin/cycle RTL vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compiler'))
from tinylpu_compiler.isa import IsaRuntime,Profile,compile_linear,bind,encode
from tinylpu_compiler.architecture import bank_pad
from tinylpu_compiler.device import Device,wire_bytes,ENABLE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capacity',type=int,default=128)
    parser.add_argument('--lanes',type=int,default=2)
    parser.add_argument('--output',default='reports/device')
    args=parser.parse_args()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    rng=random.Random(20261104)
    isa=IsaRuntime(ROOT/'build_esl/libtinylpu_isa.so')
    rows=[];total_cycles=0
    vector=folder/'cycles.txt'
    with vector.open('w') as trace,Device(ROOT/'build_esl/libtinylpu_device.so',args.capacity,args.lanes) as device:
        def step(inp):
            nonlocal total_cycles
            out,event=device.tick(inp);trace.write(f'{inp:08x} {out:06x}\n');total_cycles+=1
            return out,event
        def reset():
            step(1<<12);step(1<<12);step(ENABLE)
        def transact(name,wire,expected=None,expect_error=False,stall=True,fresh_reset=True):
            if fresh_reset:reset()
            step(ENABLE|(1<<10))
            cursor=0;received=bytearray();pending_byte=False;done=False
            # Source valid/data remains asserted until a handshake, except reset/abort tests.
            for cycle in range(250000):
                if cursor<len(wire) and (pending_byte or not stall or rng.random()>.3):pending_byte=True
                inp=ENABLE
                if pending_byte:inp|=(1<<8)|wire[cursor]
                if not stall or rng.random()>.4:inp|=1<<9
                out,event=step(inp)
                if event&1:cursor+=1;pending_byte=False
                if event&2:received.append((event>>8)&255)
                if out&(64<<8):
                    assert expect_error,(name,device.debug());done=True;break
                if out&(32<<8):
                    assert not expect_error,name
                    assert cursor==len(wire) and bytes(received)==expected,(name,received.hex(),expected.hex())
                    done=True;break
            assert done,(name,'timeout',device.debug())
            debug=device.debug();rows.append({'name':name,'wire_bytes':len(wire),'accepted':cursor,'output_bytes':len(received),'debug':debug,'expected_error':expect_error})
        def direct(name,commands,loads,explicit=None):
            encoded=encode(commands)
            report=isa.run_raw(encoded,loads,Profile(args.capacity,args.lanes,3,32,1),4)
            expected=bytes.fromhex(report['stores_hex'])
            if explicit is not None:assert expected==explicit,(name,'independent integer oracle',expected,explicit)
            wire=bytearray();offset=cursor=0
            for command in commands:
                length=8 if command['op']=='MVM' else 4
                wire.extend(encoded[offset:offset+length]);offset+=length
                if command['op']=='LOAD':
                    count=command['count'];wire.extend(loads[cursor:cursor+count]);cursor+=count
            assert cursor==len(loads)
            transact(name,wire,expected)
            for field in ('macs','load_bytes','store_bytes','command_count'):
                assert rows[-1]['debug'].get('commands' if field=='command_count' else field)==report[field]
            return bytes(wire)
        def transfer(op,addr,count):return {'op':op,'addr':addr,'count':count}
        def mvm(a,w,y,m,k,stride,acc=False,sat=False):
            return {'op':'MVM','a':a,'w':w,'y':y,'rows':m,'k':k,'stride':stride,
                    'input_bits':4,'accumulate':acc,'sat16':sat}
        end={'op':'END'}
        for m,k,frames in ((1,1,1),(3,17,2),(7,16,1),(8,16,8),(8,64,2),(3,65,2),(3,256,1),(3,512,1)):
            weights=[[rng.randrange(-8,8) for _ in range(k)] for _ in range(m)]
            inputs=[[rng.randrange(-8,8) for _ in range(k)] for _ in range(frames)]
            p=Profile(args.capacity,args.lanes,3,32,1)
            for out_bits in (16,32):
                for storage in ('packed','bank4'):
                    tk=min(k,16);tm=min(m,2)
                    plan=compile_linear(weights,frames,p,output_bits=out_bits,tile_rows=tm,tile_k=tk)
                    if storage=='bank4':plan=bank_pad(plan)
                    commands,loads=bind(plan,inputs)
                    report=isa.run_raw(commands,loads,p,4)
                    expected=bytes.fromhex(report['stores_hex'])
                    name=f'm{m}_k{k}_b{frames}_o{out_bits}_{storage}'
                    transact(name,wire_bytes(plan,inputs),expected)
                    for field in ('macs','load_bytes','store_bytes','command_count'):
                        actual=rows[-1]['debug'].get('commands' if field=='command_count' else field)
                        assert actual==report[field],(name,field,actual,report[field])
        # Positive extreme then negative cancellation: only final SAT16 is allowed.
        for k in (512,1024):
            weights=[[-8]*512+([7]*512 if k==1024 else [])]*3
            p=Profile(args.capacity,args.lanes,3,32,1)
            plan=bank_pad(compile_linear(weights,profile=p,output_bits=16,tile_rows=2,tile_k=16))
            inp=[[-8]*k];commands,loads=bind(plan,inp);r=isa.run_raw(commands,loads,p,4)
            transact('extreme_'+str(k),wire_bytes(plan,inp),bytes.fromhex(r['stores_hex']))
        # Large native instructions exercise multiple groups, tails and address broadcasts.
        m=min(16,(args.capacity-16)//4)
        direct('native_rows_broadcast',[transfer('LOAD',0,16),mvm(0,0,16,m,1,1),
               transfer('STORE',16,m*4),end],bytes([0x88]*16),b'\x40\0\0\0'*m)
        direct('native_k64_alias',[transfer('LOAD',0,32),mvm(0,0,32,1,64,32),
               transfer('STORE',32,4),end],bytes([0x88]*32),(4096).to_bytes(4,'little'))
        conflict=direct('all_roles_one_bank',[transfer('LOAD',0,24),mvm(16,0,24,2,16,8),
                        transfer('STORE',24,8),end],bytes([0x88]*24),(1024).to_bytes(4,'little')*2)
        direct('a_alias_second_lane',[transfer('LOAD',0,16),mvm(8,0,16,2,16,8),
               transfer('STORE',16,8),end],bytes([0x88]*16),(1024).to_bytes(4,'little')*2)
        payload=bytes(rng.randrange(256) for _ in range(args.capacity))
        direct('full_capacity_transfer',[transfer('LOAD',0,args.capacity),
               transfer('STORE',0,args.capacity),end],payload,payload)
        # Every signed INT4 product has an arithmetic oracle outside either execution model.
        commands=[];loads=bytearray();products=bytearray()
        for a in range(-8,8):
            for w in range(-8,8):
                commands.extend([transfer('LOAD',0,1),transfer('LOAD',4,1),
                                 mvm(0,4,8,1,1,1),transfer('STORE',8,4)])
                loads.extend([a&15,w&15]);products.extend((a*w).to_bytes(4,'little',signed=True))
        direct('all_256_int4_products',commands+[end],bytes(loads),bytes(products))
        for seed,operand,sat in ((0x7fffffff,1,False),(0x80000000,15,False),
                                 (32767,1,True),(0xffff8000,15,True)):
            value=seed if seed<0x80000000 else seed-(1<<32)
            value+=1 if operand==1 else -1
            value=((value+(1<<31))%(1<<32))-(1<<31)
            if sat:value=max(-32768,min(32767,value))
            size=2 if sat else 4
            direct(f'seed_{seed:x}_{operand}_{sat}',[transfer('LOAD',0,1),transfer('LOAD',4,1),
                   transfer('LOAD',8,4),mvm(0,4,8,1,1,1,acc=True,sat=sat),
                   transfer('STORE',8,size),end],bytes([operand,1])+seed.to_bytes(4,'little'),
                   value.to_bytes(size,'little',signed=True))
        # Strict header/validity/range faults; W-end arithmetic must not wrap.
        bad=[('opcode',bytes([99,0,0,0])),('reserved_end',bytes([15,1,0,0])),
             ('uninitialized',bytes([2,0,0,0,15,0,0,0])),
             ('oversized_load',bytes([1,0,127,255])),
             ('int8_disabled',bytes([3,4,0,8,32,0,0,1])),
             ('wrapped_w_end',bytes([3,0,0,255,32,15,63,255])),
             ('y_alias',bytes([3,0,0,8,0,0,0,1])),
             ('bad_alignment',bytes([3,0,0,8,33,0,0,1]))]
        for name,wire in bad:transact(name,wire,expect_error=True)
        # Abort while the arithmetic pipeline or output handshake is live. No aborted
        # edge can be counted as RX/TX; reset/ena must invalidate all loaded scratch.
        for state in (11,7):
            for mode in ('reset','ena','start'):
                reset();step(ENABLE|(1<<10));cursor=0
                for _ in range(2000):
                    inp=ENABLE|((1<<8)|conflict[cursor] if cursor<len(conflict) else 0)
                    _,event=step(inp)
                    if event&1:cursor+=1
                    if device.debug()['state']==state:break
                else:raise AssertionError(('abort state not reached',state))
                if state==7:
                    held=step(ENABLE)[0]
                    for _ in range(9):assert step(ENABLE)[0]==held
                abort={'reset':1<<12,'ena':1<<11,'start':ENABLE|(1<<10)|(1<<9)|(1<<8)|15}[mode]
                out,event=step(abort);assert event==0
                assert (out&(64<<8)) if mode=='start' else out==0
                rows.append({'name':f'abort_{state}_{mode}','expected_error':mode=='start'})
                transact(f'abort_{state}_{mode}_recovery',bytes([15,0,0,0]),b'')
        transact('restart_done',bytes([15,0,0,0]),b'',fresh_reset=False)
        reset();step(ENABLE|(1<<10));wire=bytes([3,0,0,8,32,0,0,1]);cursor=0
        for _ in range(30):
            _,event=step(ENABLE|((1<<8)|wire[cursor] if cursor<len(wire) else 0))
            if event&1:cursor+=1
            if device.debug()['state']==15:break
        else:raise AssertionError('MVM validation pipeline state not reached')
        out,event=step(ENABLE|(1<<10)|(1<<9)|(1<<8)|15)
        assert out&(64<<8) and event==0
        rows.append({'name':'start_during_mvm_validation','expected_error':True})
        # Abort partial RX and busy start, recover after reset; source stalls do not consume bytes.
        reset();step(ENABLE|(1<<10));step(ENABLE|(1<<8)|3);step(ENABLE)
        out,_=step(ENABLE|(1<<10));assert out&(64<<8)
        out,_=step(1<<11);assert out==0
        transact('ena_recovery',bytes([15,0,0,0]),b'')
        # Holding start high has no repeated edge, partial header/payload waits.
        reset();step(ENABLE|(1<<10))
        for _ in range(5):out,_=step(ENABLE|(1<<10));assert not out&(64<<8)
        for _ in range(4):step(ENABLE)
        reset()
    summary={'cases':len(rows),'cycles':total_cycles,'all_numeric_checks_passed':True,
             'capacity':args.capacity,'lanes':args.lanes,'vectors_sha256':hashlib.sha256(vector.read_bytes()).hexdigest(),
             'scope':'C++ pin/cycle reference; independent ISA engine validates integer result and work counts', 'records':rows}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
