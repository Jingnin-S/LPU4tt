#!/usr/bin/env python3
"""Real upstream DAG/Relax VM/new ISA composition and SGLang callsite contracts."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"compiler"),str(ROOT/"scripts")]
from run_compiler_experiments import build_models
from run_isa_experiments import dump,csv_file
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.isa import Profile
from tinylpu_compiler.architecture import Resources,architecture_time
from tinylpu_compiler.composition_vm import CompositionVm
from tinylpu_compiler.sglang_adapter import TinyLpuLinearMethod


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--library",default="build_esl/libtinylpu_isa.so")
    parser.add_argument("--architecture-library",default="build_esl/libtinylpu_architecture.so")
    parser.add_argument("--output",default="reports/architecture/composition")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261104)

    class Qkv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name,phase in (("q",1),("k",2),("v",3)):
                self.register_buffer(name,torch.tensor([[(r*5+c*3+phase)%5-2 for c in range(16)] for r in range(8)],dtype=torch.float32))
        def forward(self,x):
            return tuple(torch.nn.functional.linear(x,w) for w in (self.q,self.k,self.v))
    class Ffn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("up",torch.tensor([[(r*3+c*5+1)%5-2 for c in range(16)] for r in range(24)],dtype=torch.float32))
            self.register_buffer("down",torch.tensor([[(r*5+c*3+2)%5-2 for c in range(24)] for r in range(8)],dtype=torch.float32))
        def forward(self,x):
            h=torch.nn.functional.linear(x,self.up)
            h=torch.clamp(torch.floor(torch.relu(h)*.75+.5),0,7)
            return torch.nn.functional.linear(h,self.down)
    models=build_models(torch)+[("qkv_16x8",Qkv(),16),("ffn_16x24x8",Ffn(),16)]
    presets=[("lite64",Profile(64,1,3,32,1),Resources(0,1,1,1),"packed"),
             ("flat3_128",Profile(128,2,3,32,1),Resources(0,3,1,1),"packed"),
             ("bank4_128",Profile(128,2,3,32,1),Resources(4,1,1,1),"bank4")]
    records=[]; frames_checked=0
    for name,model,k in models:
        inputs=rng.integers(-8,8,size=(16,k),dtype=np.int32).astype(np.float32)
        for frames in (1,8):
            folder=output/name/f"batch{frames}";folder.mkdir(parents=True,exist_ok=True)
            nodes,metadata=import_model(model,(torch.zeros(frames,k),),folder/"frontend",multi_output=True,allow_batch=True)
            dump(folder/"frontend.json",metadata)
            for label,profile,resources,storage in presets:
                for fusion in ((False,True) if name.startswith("qkv") else (False,)):
                    case=folder/f"{label}_fusion{int(fusion)}"
                    with CompositionVm(nodes,case,args.library,args.architecture_library,frames,profile,resources,storage,fusion) as vm:
                        samples=[]
                        for batch in (inputs[:frames],inputs[8:8+frames]):
                            actual=vm.run(batch)
                            with torch.no_grad(): expected=model(torch.from_numpy(batch))
                            actuals=actual if isinstance(actual,tuple) else (actual,)
                            expecteds=expected if isinstance(expected,tuple) else (expected,)
                            assert len(actuals)==len(expecteds)
                            assert all(np.array_equal(a,e.numpy()) for a,e in zip(actuals,expecteds))
                            samples.append({"input":batch.tolist(),"outputs":[a.tolist() for a in actuals],
                                            "reports":copy.deepcopy(vm.last_reports),"software_timing":dict(vm.last_timing)})
                            frames_checked+=frames
                        before=vm.total_callback_calls
                        for bad in (np.full((frames,k),np.nan,dtype=np.float32),inputs[:frames]+.5,inputs[:frames].astype(np.float64)):
                            try: vm.run(bad)
                            except ValueError: pass
                            else: raise AssertionError("Bad VM input accepted")
                            assert vm.last_reports==[] and vm.total_callback_calls==before
                        vm.run(inputs[:frames]) # recovery
                        rr=[p["resources"] for p in vm.partitions]
                        row={"model":name,"frames":frames,"preset":label,"fused_qkv":fusion,
                             "partitions":len(vm.partitions),"host_quant_elements":vm.host_quant_elements,
                             "command_bytes":sum(r["command_bytes"] for r in rr),"load_bytes":sum(r["load_bytes"] for r in rr),
                             "store_bytes":sum(r["store_bytes"] for r in rr),"device_cycles":sum(r["device_cycles"] for r in rr),
                             "conflict_cycles":sum(r["conflict_cycles"] for r in rr),
                             "model_us":sum(architecture_time(r) for r in rr)*1e6,
                             "with_1us_per_partition_us":sum(architecture_time(r) for r in rr)*1e6+len(rr),
                             "build_wall_ns":vm.build_wall_ns,"all_passed":True}
                        records.append(row)
                        dump(case/"validation.json",{"summary":row,"samples":samples,"negative_inputs":3})
            print(json.dumps({"model":name,"frames":frames,"vm_builds":len(records)}),flush=True)
    csv_file(output/"comparison.csv",records)

    # Run the *actual pinned SGLang forward function* in a dependency-isolated
    # callsite harness. No claim that the full package/server was imported.
    revision="baa9befa381511cd4d56f83980951e5a45fffc6d"
    url=f"https://raw.githubusercontent.com/sgl-project/sglang/{revision}/python/sglang/srt/layers/linear.py"
    research=output.parent/"research";research.mkdir(exist_ok=True)
    source_path=research/"sglang_linear.py"
    if not source_path.exists(): source_path.write_bytes(urllib.request.urlopen(url,timeout=30).read())
    source=source_path.read_bytes();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ReplicatedLinear")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),method],type_ignores=[])
    ast.fix_missing_locations(module);scope={"torch":torch};exec(compile(module,str(source_path),"exec"),scope)
    class Original:
        def apply(self,layer,x,bias=None):return torch.nn.functional.linear(x,layer.weight,bias)
    bridge=TinyLpuLinearMethod(Original(),args.library,args.architecture_library)
    sg=[]
    for frames in (1,8):
        for m,k in ((3,17),(8,16),(8,64)):
            w=torch.tensor(rng.normal(size=(m,k)).astype(np.float32));bias=torch.tensor(rng.normal(size=m).astype(np.float32))
            x=torch.tensor(rng.normal(size=(frames,k)).astype(np.float32))
            layer=SimpleNamespace(weight=w,bias=bias,skip_bias_add=False,quant_method=bridge)
            with torch.no_grad(): actual,out_bias=scope["forward"](layer,x)
            debug=copy.deepcopy(bridge.last_debug)
            aq=np.clip(np.rint(x.numpy().astype(np.float64)/np.asarray(debug["activation_scales"])[:,None]),-8,7).astype(int)
            wq=np.clip(np.rint(w.numpy().astype(np.float64)/np.asarray(debug["weight_scales"])[:,None]),-8,7).astype(int)
            raw=np.asarray([[sum(int(a)*int(b) for a,b in zip(vector,row)) for row in wq] for vector in aq])
            assert raw.tolist()==debug["raw_integer_output"]
            deq=(raw*np.asarray(debug["activation_scales"])[:,None]*np.asarray(debug["weight_scales"])[None,:]).astype(np.float32)
            assert np.array_equal(actual.numpy(),(torch.from_numpy(deq)+bias).numpy()) and out_bias is None
            reference=torch.nn.functional.linear(x,w,bias).numpy()
            nrmse=float(np.linalg.norm(actual.numpy()-reference)/np.linalg.norm(reference))
            layer.skip_bias_add=True
            with torch.no_grad(): no_bias,returned_bias=scope["forward"](layer,x)
            assert returned_bias is bias and torch.equal(no_bias,torch.from_numpy(deq))
            sg.append({"frames":frames,"m":m,"k":k,"nrmse_vs_fp32":nrmse,"debug":debug,"skip_bias_contract_passed":True})
    dump(output/"sglang_contract.json",{"source_url":url,"revision":revision,"source_sha256":hashlib.sha256(source).hexdigest(),
         "scope":"actual ReplicatedLinear.forward AST; isolated CPU apply interposer; full SRT not imported or served",
         "samples":sg,"all_quantized_checks_passed":True})
    summary={"vm_combinations":len(records),"validated_input_frames":frames_checked,"all_passed":True,
             "sglang_shapes":len(sg),"sglang_full_service":False,"new_isa_vm":True}
    dump(output/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
