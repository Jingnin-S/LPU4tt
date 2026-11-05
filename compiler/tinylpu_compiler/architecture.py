"""Port-aware target selection and actual bank-padded ISA lowering."""
import copy
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .isa import Profile, compile_linear, encode, predicted


@dataclass(frozen=True)
class Resources:
    banks: int = 4
    flat_read_ports: int = 3
    read_latency: int = 1
    copy_bytes_per_cycle: int = 1


class ArchRuntime:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.tlpu_arch_analyze.argtypes = [ctypes.c_void_p, ctypes.c_size_t]+[ctypes.c_uint]*8
        self.library.tlpu_arch_analyze.restype = ctypes.c_char_p
        self.library.tlpu_arch_last_error.restype = ctypes.c_char_p
        self.library.tlpu_arch_last_error.argtypes = []

    def analyze(self, plan, resources):
        p = Profile(**plan["profile"])
        data = bytes.fromhex(plan["commands_hex"])
        value = self.library.tlpu_arch_analyze(data, len(data), plan["encoding"], p.lanes, p.depth,
                                             p.accumulator_bits, resources.banks, resources.flat_read_ports,
                                             resources.read_latency, resources.copy_bytes_per_cycle)
        if value is None:
            raise ValueError(self.library.tlpu_arch_last_error().decode())
        return json.loads(value)


def padded_stride(row_bytes):
    return row_bytes+(2-row_bytes) % 4


def bank_layout(rows, tile_k, global_k, bits, output_bits, profile):
    stride = padded_stride(math.ceil(tile_k*bits/8))
    w_size = math.ceil(rows/profile.lanes)*profile.lanes*stride
    a = w_size+(1-w_size) % 4
    a_size = math.ceil(tile_k*bits/8)
    unit = max(output_bits, profile.accumulator_bits if global_k > tile_k else output_bits)//8
    y = math.ceil((a+a_size)/unit)*unit
    end = y+rows*unit
    if end > profile.scratch_bytes:
        raise ValueError("Bank-padded tile exceeds actual capacity")
    return {"w": 0, "a": a, "y": y, "end": end, "w_bytes": w_size,
            "a_bytes": a_size, "y_bytes": rows*unit, "stride": stride}


def bank_pad(plan):
    result = copy.deepcopy(plan)
    p = Profile(**plan["profile"])
    layout = bank_layout(plan["tile_rows"], plan["tile_k"], plan["k"], plan["input_bits"], plan["output_bits"], p)
    blob, old_blob, chunks = bytearray(), bytes.fromhex(plan["weights_hex"]), {}
    for spec in result["load_specs"]:
        command = result["commands"][spec["command"]]
        if spec["source"] == "input":
            command["addr"] = layout["a"]
            continue
        mvm = next(c for c in result["commands"][spec["command"]+1:] if c["op"] == "MVM")
        row_bytes = math.ceil(mvm["k"]*plan["input_bits"]/8)
        stride = padded_stride(row_bytes)
        key = (spec["offset"], spec["count"], stride)
        if key not in chunks:
            start = len(blob)
            old = old_blob[spec["offset"]:spec["offset"]+spec["count"]]
            for row in range(mvm["rows"]):
                blob.extend(old[row*row_bytes:(row+1)*row_bytes])
                blob.extend(bytes(stride-row_bytes))
            chunks[key] = (start, len(blob)-start)
        spec["offset"], spec["count"] = chunks[key]
        command["addr"], command["count"] = layout["w"], spec["count"]
    for command in result["commands"]:
        if command["op"] == "MVM":
            command.update(a=layout["a"], w=layout["w"], y=layout["y"],
                           stride=padded_stride(math.ceil(command["k"]*plan["input_bits"]/8)))
        elif command["op"] == "STORE":
            command["addr"] = layout["y"]
    result["layout"] = layout
    result["weights_hex"] = blob.hex()
    data = encode(result["commands"], result["encoding"])
    result["commands_hex"] = data.hex()
    result["commands_sha256"] = hashlib.sha256(data).hexdigest()
    result["predicted"] = predicted(result["commands"], p, result["encoding"])
    result["storage_policy"] = "bank4: W stride=2 mod4, A base=1 mod4"
    return result


def architecture_time(report, hz=100e6, link_Bps=10e6, host_ns=0):
    return (report["device_cycles"]/hz +
            (report["command_bytes"]+report["load_bytes"]+report["store_bytes"])/link_Bps+
            report["command_count"]*host_ns*1e-9)


def choose_plan(weights, frames, profile, resources, analyzer, storage="packed", output_bits=16):
    m, k = len(weights), len(weights[0])
    best, candidates = None, []
    for rows in range(1, min(m, 16)+1):
        for tk in sorted({min(k, 64), *[n for n in (4, 8, 16, 32, 48, 64) if n <= min(k, 64)]}):
            for order in ("frame_major", "weight_stationary"):
                if storage == "bank4":
                    try:
                        bank_layout(rows, tk, k, 4, output_bits, profile)
                    except ValueError:
                        continue
                try:
                    plan = compile_linear(weights, frames, profile, 4, output_bits, 4, rows, tk, order)
                except ValueError as error:
                    if "fit scratch" in str(error):
                        continue
                    raise
                if storage == "bank4":
                    plan = bank_pad(plan)
                elif storage != "packed":
                    raise ValueError("Unknown storage policy")
                report = analyzer.analyze(plan, resources)
                cost = architecture_time(report)
                candidates.append({"rows": rows, "tile_k": tk, "order": order, "time_us": cost*1e6,
                                   "conflict_cycles": report["conflict_cycles"], "peak_bytes": plan["layout"]["end"]})
                key = (cost, report["command_bytes"], plan["layout"]["end"], rows, tk, order)
                if best is None or key < best[0]:
                    best = key, plan, report
    if best is None:
        raise ValueError("No feasible architecture tile")
    return best[1], best[2], candidates
