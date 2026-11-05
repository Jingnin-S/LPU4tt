import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"compiler"))

from tinylpu_compiler.isa import (IsaRuntime, Profile, bind, compile_linear, decode,
                                 encode, estimate_shape, layout, oracle, pack, time_seconds)


def test_encoding_roundtrip_and_actual_bytes():
    commands = [{"op": "LOAD", "addr": 0, "count": 256},
                {"op": "MVM", "a": 0, "w": 8, "y": 24, "rows": 2, "k": 17,
                 "stride": 9, "input_bits": 4, "accumulate": True, "sat16": True},
                {"op": "STORE", "addr": 24, "count": 4}, {"op": "END"}]
    for encoding, length in ((4, 20), (8, 32)):
        data = encode(commands, encoding)
        assert len(data) == length
        assert decode(data, encoding) == commands
    assert encode(commands, 4)[4:12] == bytes([3, 3, 0, 8, 24, 1, 16, 9])


@pytest.mark.parametrize("data", [b"", b"\x00\x00\x00\x00", b"\x0f", b"\x0f\x01\x00\x00",
                                  b"\x0f\x00\x00\x00"*2, bytes([3, 8, 0, 0, 16, 0, 0, 1])])
def test_bad_encodings(data):
    with pytest.raises(ValueError):
        decode(data)


def test_memory_layout_and_no_silent_float_conversion():
    assert pack([-8, 7, -1], 4) == b"\x78\x0f"
    assert layout(8, 16, 16, 4, 16, Profile(96))["end"] == 88
    assert layout(8, 16, 16, 4, 32, Profile(128))["end"] == 104
    with pytest.raises(ValueError):
        layout(8, 16, 16, 4, 32, Profile(96))
    for values in ([1.0], [True], [8], [-9]):
        with pytest.raises(ValueError):
            pack(values, 4)


def test_long_k_clear_accumulate_and_final_sat():
    plan = compile_linear([[-8]*65]*3, profile=Profile(64), tile_rows=2, tile_k=16, output_bits=16)
    mvms = [c for c in plan["commands"] if c["op"] == "MVM"]
    assert [c["accumulate"] for c in mvms] == [False, True, True, True, True]*2
    assert [c["sat16"] for c in mvms] == [False]*4+[True]+[False]*4+[True]
    assert [c["k"] for c in mvms] == [16]*4+[1]+[16]*4+[1]
    assert len(bind(plan, [[-8]*65])[1]) == plan["predicted"]["load_bytes"]
    assert oracle([[-8]*512], [[-8]*512], 32, 16) == ([[32767]], 0, 1)
    assert oracle([[-8]*512], [[-8]*512], 16, 16) == ([[-32768]], 1, 0)


def test_residency_orders_and_binding_is_runtime_only():
    kwargs = dict(weights=[[1]*16]*8, frames=8, profile=Profile(96), tile_rows=4, tile_k=16)
    first = compile_linear(**kwargs, order="frame_major")
    ws = compile_linear(**kwargs, order="weight_stationary")
    assert first["predicted"]["load_bytes"] > ws["predicted"]["load_bytes"]
    a, b = bind(ws, [[1]*16]*8), bind(ws, [[-8]*16]*8)
    assert a[0] == b[0] and a[1] != b[1]
    mixed = compile_linear(**kwargs, encoding=4)
    fixed = compile_linear(**kwargs, encoding=8)
    for field in mixed["predicted"]:
        if field != "command_bytes":
            assert mixed["predicted"][field] == fixed["predicted"][field]
    assert time_seconds(mixed["predicted"]) < time_seconds(fixed["predicted"])


def test_shape_and_capability_guards():
    with pytest.raises(ValueError):
        compile_linear([[1]*16], input_bits=8)
    with pytest.raises(ValueError):
        compile_linear([[1]*4097])
    with pytest.raises(ValueError):
        compile_linear([[1]*16], profile=Profile(accumulator_bits=16), output_bits=32)
    estimate = compile_linear([[1]*16], materialize=False)
    with pytest.raises(ValueError):
        bind(estimate, [[1]*16])
    with pytest.raises(ValueError):
        bind(compile_linear([[1]*16]), [[1.0]*16])


def test_search_estimator_matches_emitted_stream_with_odd_tails():
    for k in (1, 17, 65):
        for frames in (1, 3):
            for order in ("frame_major", "weight_stationary"):
                for output in (16, 32):
                    for tm, tk in ((1, 1), (2, min(k, 8)), (3, min(k, 16))):
                        profile = Profile(128)
                        plan = compile_linear([[1]*k]*7, frames, profile, 4, output, 4, tm, tk, order)
                        assert estimate_shape(7, k, frames, tm, tk, 4, output, profile, order) == plan["predicted"]


def runtime():
    default = Path("build_esl/libtinylpu_isa.so")
    path = Path(os.environ.get("TINY_LPU_ISA_LIBRARY", str(default)))
    if not path.exists():
        pytest.skip("C++ ISA shared library unavailable on this platform")
    return IsaRuntime(path)


@pytest.mark.parametrize("encoding", [4, 8])
def test_cpp_compiled_program_matches_independent_math(encoding):
    rt = runtime()
    for bits, k in ((4, 17), (4, 65), (4, 512), (8, 65)):
        weights = [[-(1 << (bits-1))]*k for _ in range(3)]
        inputs = [[-(1 << (bits-1))]*k, [1]*k]
        for accumulator in (16, 32):
            for output in sorted({16, accumulator}):
                plan = compile_linear(weights, frames=2, profile=Profile(64, accumulator_bits=accumulator, enable_int8=True),
                                      input_bits=bits, output_bits=output, tile_rows=2, tile_k=8, encoding=encoding)
                result, report = rt.run(plan, inputs)
                expected, overflows, saturations = oracle(weights, inputs, accumulator, output)
                assert result == expected
                assert report["overflow_events"] == overflows
                assert report["saturation_events"] == saturations


def test_cpp_rejects_and_recovers():
    rt = runtime()
    for commands, loads in ((b"\x0f", b""), (encode([{"op": "END"}]), b"\x00"),
                            (encode([{"op": "STORE", "addr": 0, "count": 4}, {"op": "END"}]), b"")):
        with pytest.raises(ValueError):
            rt.run_raw(commands, loads)
        assert rt.run_raw(encode([{"op": "END"}]), b"")["command_count"] == 1


def test_bias_seed_and_final_saturation_are_explicit():
    rt = runtime()
    commands = [{"op": "LOAD", "addr": 0, "count": 1},
                {"op": "LOAD", "addr": 8, "count": 2},
                {"op": "LOAD", "addr": 16, "count": 8},
                {"op": "MVM", "a": 0, "w": 8, "y": 16, "rows": 2, "k": 2, "stride": 1,
                 "input_bits": 4, "accumulate": True, "sat16": True},
                {"op": "STORE", "addr": 16, "count": 4}, {"op": "END"}]
    payload = pack([-8, -8], 4)+pack([-8, -8], 4)+pack([7, 7], 4)
    payload += (32760).to_bytes(4, "little", signed=True)+(-32760).to_bytes(4, "little", signed=True)
    report = rt.run_raw(encode(commands), payload)
    result = bytes.fromhex(report["stores_hex"])
    assert [int.from_bytes(result[i:i+2], "little", signed=True) for i in (0, 2)] == [32767, -32768]
    assert report["saturation_events"] == 2


def test_graph_adapter_preserves_legacy_numeric_type(tmp_path):
    from tinylpu_compiler.ir import gemv, input_vector, relu_q
    from tinylpu_compiler.isa_graph import IsaGraph
    graph = IsaGraph(gemv(relu_q(gemv(input_vector(16), [[1]*16]*4)), [[1]*4]*2), tmp_path)
    assert len(graph.plans) == 2
    with pytest.raises(ValueError, match="INT16 output"):
        IsaGraph(gemv(input_vector(512), [[-8]*512]), tmp_path/"unsafe")
