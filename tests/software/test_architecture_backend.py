import os
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"compiler"))
from tinylpu_compiler.isa import IsaRuntime, Profile, compile_linear, oracle
from tinylpu_compiler.architecture import ArchRuntime, Resources, bank_pad, bank_layout


def libraries():
    paths = [Path(os.environ.get("TINY_LPU_ISA_LIBRARY", str(ROOT/"build_esl/libtinylpu_isa.so"))),
             Path(os.environ.get("TINY_LPU_ARCH_LIBRARY", str(ROOT/"build_esl/libtinylpu_architecture.so")))]
    if not all(p.exists() for p in paths):
        pytest.skip("Server ISA and architecture libraries required")
    return paths


def test_bank_layout_proof_for_all_byte_positions():
    for bits in (4, 8):
        for k in (1, 17, 64):
            p = Profile(256, memory_byte_bw=1, enable_int8=bits == 8)
            layout = bank_layout(2, k, k, bits, 32, p)
            for col in range(k):
                offset = col*bits//8
                banks = {(layout['a']+offset) % 4, offset % 4, (layout['stride']+offset) % 4}
                assert len(banks) == 3


@pytest.mark.parametrize("k", [1, 17, 65, 256])
def test_bank_padding_executes_exact_tail_and_partial_sum(k):
    isa, arch = libraries()
    p = Profile(128, memory_byte_bw=1)
    weights = [[(i*3+r*5) % 16-8 for i in range(k)] for r in range(7)]
    inputs = [[(i*7+f) % 16-8 for i in range(k)] for f in range(2)]
    plan = bank_pad(compile_linear(weights, 2, p, output_bits=32, tile_rows=2, tile_k=min(k, 16)))
    actual, report = IsaRuntime(isa).run(plan, inputs)
    assert actual == oracle(weights, inputs, 32, 32)[0]
    resources = ArchRuntime(arch).analyze(plan, Resources())
    assert resources["conflict_cycles"] == 0
    assert resources["load_bytes"] == report["load_bytes"]
    assert resources["core_cycles"] > report["core_cycles"]  # registered memory read


def test_new_vm_branch_fusion_and_recovery(tmp_path):
    import numpy as np
    isa, arch = libraries()
    pytest.importorskip("tvm")
    from tinylpu_compiler.ir import input_vector, gemv
    from tinylpu_compiler.composition_vm import CompositionVm
    x = input_vector(16)
    nodes = tuple(gemv(x, [[i+1]*16]*3) for i in range(3))
    values = np.arange(16, dtype=np.float32).reshape(1, 16) % 8
    results = []
    for fusion in (False, True):
        with CompositionVm(nodes, tmp_path/str(fusion), isa, arch, fuse_qkv=fusion) as vm:
            actual = vm.run(values)
            assert len(vm.last_reports) == (1 if fusion else 3)
            assert all(np.array_equal(a, np.full((1, 3), values.sum()*(i+1))) for i, a in enumerate(actual))
            results.append(actual)
            with pytest.raises(ValueError):
                vm.run(values+.5)
            assert vm.last_reports == [] and vm.last_timing == {}
            assert len(vm.run(values)) == 3
        with pytest.raises(RuntimeError):
            vm.run(values)
    assert all(np.array_equal(a, b) for a, b in zip(*results))


def test_sglang_linear_bridge_exact_quantized_reference_and_fallback():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from tinylpu_compiler.sglang_adapter import TinyLpuLinearMethod
    isa, arch = libraries()
    class Original:
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)
    adapter = TinyLpuLinearMethod(Original(), isa, arch)
    layer = SimpleNamespace(weight=torch.tensor([[.2, -.3, .1, .9], [-.7, .2, .3, -.1]]))
    x = torch.tensor([[.1, .2, -.3, .4]])
    output = adapter.apply(layer, x)
    debug = adapter.last_debug
    aq = np.rint(x.numpy()/np.array(debug["activation_scales"])[:, None]).clip(-8, 7).astype(int)
    wq = np.rint(layer.weight.numpy()/np.array(debug["weight_scales"])[:, None]).clip(-8, 7).astype(int)
    raw = [[sum(int(a)*int(w) for a,w in zip(aq[0], row)) for row in wq]]
    assert raw == debug["raw_integer_output"]
    expected = (np.asarray(raw)*np.array(debug["activation_scales"])[:, None]*np.array(debug["weight_scales"])[None,:]).astype(np.float32)
    assert np.array_equal(output.numpy(), expected)
    layer.weight[0, 0] += .8
    adapter.apply(layer, x)
    assert len(adapter.cache) == 2
    layer.weight = layer.weight.double()
    actual = adapter.apply(layer, x.double())
    assert adapter.fallback_calls == 1
    assert torch.equal(actual, torch.nn.functional.linear(x.double(), layer.weight))
    layer.weight = layer.weight.float()
    layer.use_intel_amx_backend = True
    assert torch.equal(adapter.apply(layer, x), torch.nn.functional.linear(x, layer.weight))
    assert adapter.last_debug['reason'] == 'packed CPU weights require the original method'
    layer.use_intel_amx_backend = False
    layer.weight = torch.empty((2, 0), dtype=torch.float32)
    assert torch.equal(adapter.apply(layer, torch.empty((1, 0))), torch.zeros((1, 2)))
    assert adapter.fallback_calls == 3
