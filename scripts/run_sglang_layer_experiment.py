#!/usr/bin/env python3
"""Optional integration with a normally constructed, installed SGLang CPU layer.

Requires a compatible SGLang CPU Python environment. Does not start SRT serving.
"""
import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"compiler"))
from tinylpu_compiler.sglang_adapter import attach_linear
from tinylpu_compiler.isa import Profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/architecture/sglang_actual')
    parser.add_argument('--capacity',type=int,default=128)
    parser.add_argument('--lanes',type=int,default=2)
    parser.add_argument('--engine',choices=('isa','device'),default='isa')
    args=parser.parse_args()
    from sglang.srt.layers.linear import ReplicatedLinear
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261104)
    rows=[]
    for frames in (1,8):
        for m,k in ((3,17),(8,16),(8,64)):
            layer=ReplicatedLinear(k,m,bias=True,params_dtype=torch.float32)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(rng.normal(size=(m,k)).astype(np.float32)))
                layer.bias.copy_(torch.from_numpy(rng.normal(size=m).astype(np.float32)))
                layer.quant_method.process_weights_after_loading(layer)
                x=torch.from_numpy(rng.normal(size=(frames,k)).astype(np.float32))
                reference=torch.nn.functional.linear(x,layer.weight,layer.bias)
                attach_linear(layer,ROOT/'build_esl/libtinylpu_isa.so',ROOT/'build_esl/libtinylpu_architecture.so',
                              profile=Profile(args.capacity,args.lanes,3,32,1),
                              device_library=ROOT/'build_esl/libtinylpu_device.so' if args.engine=='device' else None)
                actual,returned_bias=layer(x)
                debug=layer.quant_method.last_debug
                assert returned_bias is None and layer.quant_method.offload_calls==1
                aq=np.rint(x.numpy().astype(np.float64)/np.asarray(debug['activation_scales'])[:,None]).clip(-8,7).astype(int)
                wq=np.rint(layer.weight.numpy().astype(np.float64)/np.asarray(debug['weight_scales'])[:,None]).clip(-8,7).astype(int)
                raw=np.asarray([[sum(int(a)*int(b) for a,b in zip(vector,row)) for row in wq] for vector in aq])
                assert raw.tolist()==debug['raw_integer_output']
                expected=(raw*np.asarray(debug['activation_scales'])[:,None]*np.asarray(debug['weight_scales'])[None,:]).astype(np.float32)
                assert torch.equal(actual,torch.from_numpy(expected)+layer.bias)
                layer.skip_bias_add=True
                no_bias,returned_bias=layer(x)
                assert returned_bias is layer.bias and torch.equal(no_bias,torch.from_numpy(expected))
                rows.append({'frames':frames,'m':m,'k':k,'offload_calls':layer.quant_method.offload_calls,
                             'nrmse_vs_fp32':float((actual-reference).norm()/reference.norm()),
                             'numeric_oracle':'PASS','bias_contract':'PASS','debug':debug})
            print(json.dumps({'actual_sglang_layer':[frames,m,k],'passed':True}),flush=True)
    result={'sglang_version':importlib.metadata.version('sglang'),'engine':args.engine,
            'profile':{'capacity':args.capacity,'lanes':args.lanes,'accumulator_bits':32},
            'class':'sglang.srt.layers.linear.ReplicatedLinear','normal_constructor':True,
            'actual_module_import':True,'shape_cases':len(rows),'full_srt_service':False,'all_passed':True,'cases':rows}
    dist=importlib.metadata.distribution('sglang')
    result['linear_source_sha256']=hashlib.sha256(Path(dist.locate_file('sglang/srt/layers/linear.py')).read_bytes()).hexdigest()
    result['post_load_hook_executed']=True
    result['runtime_versions']={name:importlib.metadata.version(name) for name in ('torch','numpy','transformers','apache-tvm-ffi')}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
