#!/usr/bin/env python3
"""Small real Relax/ISA/banked-memory integration check for a clean checkout."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"compiler"))
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.composition_vm import CompositionVm


class Qkv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[(r*5+c*3)%5-2 for c in range(16)] for r in range(3)],dtype=torch.float32))
        self.register_buffer("w2",-self.w.clone())
    def forward(self,x):
        return (torch.nn.functional.linear(x,self.w),torch.nn.functional.linear(x,self.w2))


def main():
    model=Qkv().eval()
    x=np.arange(16,dtype=np.float32).reshape(1,16)%8
    with tempfile.TemporaryDirectory(prefix="tinylpu-smoke-") as folder:
        directory=Path(folder)
        nodes,metadata=import_model(model,(torch.zeros(1,16),),directory/"frontend",multi_output=True)
        with CompositionVm(nodes,directory/"vm",ROOT/"build_esl/libtinylpu_isa.so",
                           ROOT/"build_esl/libtinylpu_architecture.so",fuse_qkv=True) as vm:
            actual=vm.run(x)
            with torch.no_grad():expected=model(torch.from_numpy(x))
            assert all(np.array_equal(a,e.numpy()) for a,e in zip(actual,expected))
            assert len(vm.last_reports)==1
            assert vm.last_reports[0]["resources"]["conflict_cycles"]==0
    print("Clean checkout smoke PASS: torch.export -> Relax DAG -> fused VM callback -> real ISA C++")


if __name__=="__main__":main()
