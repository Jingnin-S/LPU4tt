from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compiler'))
from tinylpu_compiler.isa import IsaRuntime,Profile,compile_linear,oracle
from tinylpu_compiler.architecture import bank_pad
from tinylpu_compiler.device import DeviceRuntime,no_stall_cycles


@pytest.mark.parametrize('capacity,lanes',[(64,1),(64,2),(128,2),(256,2)])
@pytest.mark.parametrize('bits',[16,32])
def test_device_runtime_compiler_stream(capacity,lanes,bits):
    p=Profile(capacity,lanes,3,32,1)
    weights=[[-8]*65,[7]*65,[i%16-8 for i in range(65)]]
    inputs=[[-8]*65,[7]*65]
    plan=bank_pad(compile_linear(weights,2,p,output_bits=bits,tile_rows=2,tile_k=16))
    actual,report=DeviceRuntime(ROOT/'build_esl/libtinylpu_device.so').run(plan,inputs)
    independent,reference=IsaRuntime(ROOT/'build_esl/libtinylpu_isa.so').run(plan,inputs)
    assert actual==independent==oracle(weights,inputs,32,bits)[0]
    assert report['device_cycles']==no_stall_cycles(plan,lanes)['cycles']
    assert all(report[k]==reference[k] for k in ('command_count','load_bytes','store_bytes','macs','stores_hex'))


@pytest.mark.parametrize('p',[Profile(128,2,3,16,1),Profile(128,4,3,32,1),
                              Profile(128,2,2,32,1),Profile(128,2,3,32,2)])
def test_device_runtime_rejects_unsupported_profile(p):
    plan=compile_linear([[1]],profile=p,output_bits=16)
    with pytest.raises(ValueError,match='requires INT4/ACC32'):
        DeviceRuntime(ROOT/'build_esl/libtinylpu_device.so').run(plan,[[1]])
