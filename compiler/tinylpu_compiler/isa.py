"""Device ISA candidates: real encoding, static lowering, and C++ execution binding.

This is independent of the host ABI-v1 tensor plan. No floating point execution.
"""

import ctypes
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


OPS = {"LOAD": 1, "STORE": 2, "MVM": 3, "END": 15}


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def encode(commands, encoding=4):
    integer(encoding, 4, 8, "encoding")
    if encoding not in (4, 8):
        raise ValueError("encoding must be 4 or 8")
    out = bytearray()
    for command in commands:
        op = command["op"]
        if op in ("LOAD", "STORE"):
            row = [OPS[op], 0, integer(command["addr"], 0, 255, "address"),
                   integer(command["count"], 1, 256, "count")-1]
            if encoding == 8:
                row += [0]*4
        elif op == "MVM":
            bits = command["input_bits"]
            if bits not in (4, 8):
                raise ValueError("input_bits must be 4 or 8")
            flags = (int(command["accumulate"]) | int(command["sat16"]) << 1
                     | (bits == 8) << 2)
            row = [3, flags, integer(command["a"], 0, 255, "a"),
                   integer(command["w"], 0, 255, "w"),
                   integer(command["y"], 0, 255, "y"),
                   integer(command["rows"], 1, 16, "rows")-1,
                   integer(command["k"], 1, 64, "k")-1,
                   integer(command["stride"], 1, 255, "stride")]
        elif op == "END":
            row = [15]+[0]*(encoding-1)
        else:
            raise ValueError("Unknown opcode")
        out.extend(row)
    # Roundtrip also checks END ordering and reserved bits.
    decode(bytes(out), encoding)
    return bytes(out)


def decode(data, encoding=4):
    if encoding not in (4, 8):
        raise ValueError("encoding must be 4 or 8")
    commands, offset, ended = [], 0, False
    while offset < len(data):
        if ended:
            raise ValueError("Trailing commands after END")
        op = data[offset]
        length = 8 if op == 3 else encoding
        row = data[offset:offset+length]
        if len(row) != length:
            raise ValueError("Truncated command")
        if op in (1, 2):
            if row[1] or any(row[4:]):
                raise ValueError("Reserved LOAD/STORE fields")
            command = {"op": "LOAD" if op == 1 else "STORE", "addr": row[2], "count": row[3]+1}
        elif op == 3:
            if row[1] & 0xf0 or ((row[1] >> 2) & 3) > 1 or row[5] > 15 or row[6] > 63 or row[7] == 0:
                raise ValueError("Reserved or invalid MVM fields")
            command = {"op": "MVM", "a": row[2], "w": row[3], "y": row[4],
                       "rows": row[5]+1, "k": row[6]+1, "stride": row[7],
                       "input_bits": 8 if row[1] & 4 else 4,
                       "accumulate": bool(row[1] & 1), "sat16": bool(row[1] & 2)}
        elif op == 15:
            if any(row[1:]):
                raise ValueError("Reserved END fields")
            command, ended = {"op": "END"}, True
        else:
            raise ValueError("Unknown opcode")
        commands.append(command)
        offset += length
    if not ended:
        raise ValueError("Missing END")
    return commands


def pack(values, bits):
    limit = 1 << (bits-1)
    if any(isinstance(v, bool) or not hasattr(v, "__index__") for v in values):
        raise ValueError("Operands must be integer values")
    values = [integer(int(v), -limit, limit-1, "operand") for v in values]
    if bits == 8:
        return bytes(v & 255 for v in values)
    if bits != 4:
        raise ValueError("Unsupported packing")
    result = bytearray((len(values)+1)//2)
    for i, value in enumerate(values):
        result[i//2] |= (value & 15) << (4*(i % 2))
    return bytes(result)


@dataclass(frozen=True)
class Profile:
    scratch_bytes: int = 128
    lanes: int = 2
    depth: int = 3
    accumulator_bits: int = 32
    memory_byte_bw: int = 2
    enable_int8: bool = False

    def validate(self):
        integer(self.scratch_bytes, 32, 256, "scratch_bytes")
        integer(self.depth, 2, 4, "depth")
        integer(self.memory_byte_bw, 1, 16, "memory_byte_bw")
        if self.lanes not in (1, 2, 4) or self.accumulator_bits not in (16, 32):
            raise ValueError("Unsupported lane/accumulator profile")


def layout(rows, tile_k, global_k, bits, output_bits, profile):
    """Reserve padded W rows, one A slot and one partial/final Y region."""
    profile.validate()
    w_bytes = math.ceil(rows/profile.lanes)*profile.lanes*math.ceil(tile_k*bits/8)
    a_bytes = math.ceil(tile_k*bits/8)
    y_unit = max(output_bits, profile.accumulator_bits if tile_k < global_k else output_bits)//8
    y_addr = math.ceil((w_bytes+a_bytes)/y_unit)*y_unit
    end = y_addr+rows*y_unit
    if end > profile.scratch_bytes:
        raise ValueError("Tile does not fit scratch")
    return {"w": 0, "a": w_bytes, "y": y_addr, "end": end,
            "w_bytes": w_bytes, "a_bytes": a_bytes, "y_bytes": rows*y_unit}


def predicted(commands, profile, encoding):
    counts = dict(command_count=len(commands), command_bytes=len(encode(commands, encoding)),
                  mvm_count=0, macs=0, core_cycles=0, scratch_cycles=0,
                  decode_cycles=len(commands), load_bytes=0, store_bytes=0)
    for cmd in commands:
        if cmd["op"] in ("LOAD", "STORE"):
            counts["load_bytes" if cmd["op"] == "LOAD" else "store_bytes"] += cmd["count"]
            counts["scratch_cycles"] += math.ceil(cmd["count"]/profile.memory_byte_bw)
        if cmd["op"] == "MVM":
            counts["mvm_count"] += 1
            counts["macs"] += cmd["rows"]*cmd["k"]
            for start in range(0, cmd["rows"], profile.lanes):
                active = min(profile.lanes, cmd["rows"]-start)
                counts["core_cycles"] += cmd["k"]+profile.depth-1
                if cmd["accumulate"]:
                    counts["scratch_cycles"] += math.ceil(active*profile.accumulator_bits/8/profile.memory_byte_bw)
                out_bits = 16 if cmd["sat16"] else profile.accumulator_bits
                counts["scratch_cycles"] += math.ceil(active*out_bits/8/profile.memory_byte_bw)
    counts["device_cycles"] = counts["decode_cycles"]+counts["core_cycles"]+counts["scratch_cycles"]
    return counts


def time_seconds(counts, core_hz=100e6, link_Bps=10e6, host_ns=0):
    if core_hz <= 0 or link_Bps <= 0 or host_ns < 0:
        raise ValueError("Invalid performance scenario")
    return (counts["device_cycles"]/core_hz
            + (counts["command_bytes"]+counts["load_bytes"]+counts["store_bytes"])/link_Bps
            + counts["command_count"]*host_ns*1e-9)


def estimate_shape(rows, k, frames, tile_rows, tile_k, input_bits, output_bits,
                   profile, order="frame_major", encoding=4):
    """Closed-form search estimator, cross-checked against emitted/decoded streams.

    W/A regions are disjoint and invariant; residency changes only with source keys.
    This function does not encode or execute a program.
    """
    layout(tile_rows, tile_k, k, input_bits, output_bits, profile)
    row_tiles = [min(tile_rows, rows-start) for start in range(0, rows, tile_rows)]
    ks = [min(tile_k, k-start) for start in range(0, k, tile_k)]
    rb = [math.ceil(size*input_bits/8) for size in ks]
    bw, acc, out = profile.memory_byte_bw, profile.accumulator_bits//8, output_bits//8
    nrow, nk = len(row_tiles), len(ks)
    if nk > 1:
        nwl = nal = frames*nrow*nk
        wbytes = frames*rows*sum(rb)
        abytes = frames*nrow*sum(rb)
        copy = frames*sum(math.ceil(r*b/bw) for r in row_tiles for b in rb)
        copy += frames*nrow*sum(math.ceil(b/bw) for b in rb)
    else:
        w_repeats = 1 if order == "weight_stationary" or nrow == 1 else frames
        a_repeats = 1 if order == "frame_major" or frames == 1 else nrow
        nwl, nal = w_repeats*nrow, a_repeats*frames
        wbytes, abytes = w_repeats*rows*rb[0], nal*rb[0]
        copy = w_repeats*sum(math.ceil(r*rb[0]/bw) for r in row_tiles)+nal*math.ceil(rb[0]/bw)
    store_count, mvm_count = frames*nrow, frames*nrow*nk
    copy += frames*sum(math.ceil(r*out/bw) for r in row_tiles)
    active = [min(profile.lanes, r-start) for r in row_tiles for start in range(0, r, profile.lanes)]
    core = frames*len(active)*(k+nk*(profile.depth-1))
    partial = frames*(nk-1)*sum(math.ceil(a*acc/bw) for a in active)
    writes = partial+frames*sum(math.ceil(a*out/bw) for a in active)
    count = nwl+nal+store_count+mvm_count+1
    command_bytes = 8*mvm_count+encoding*(count-mvm_count)
    scratch = copy+partial+writes
    return dict(command_count=count, command_bytes=command_bytes, mvm_count=mvm_count,
                macs=frames*rows*k, core_cycles=core, scratch_cycles=scratch,
                decode_cycles=count, load_bytes=wbytes+abytes, store_bytes=frames*rows*out,
                device_cycles=count+core+scratch)


def compile_linear(weights, frames=1, profile=None, input_bits=4, output_bits=32,
                   encoding=4, tile_rows=None, tile_k=None, order="frame_major",
                   reuse_weights=True, name="linear", materialize=True):
    """Static shape/weight lowering. Runtime A values never affect the schedule."""
    profile = profile or Profile()
    profile.validate()
    integer(frames, 1, 32, "frames")
    if input_bits not in (4, 8) or (input_bits == 8 and not profile.enable_int8):
        raise ValueError("Input format not supported by profile")
    if output_bits not in (16, profile.accumulator_bits):
        raise ValueError("Output must be SAT16 or raw accumulator width")
    if order not in ("frame_major", "weight_stationary"):
        raise ValueError("Unknown order")
    # Reject float/bool operands, including numpy floats, before int conversion.
    weights = [list(row) for row in weights]
    if not weights or not weights[0]:
        raise ValueError("Empty weights")
    rows, k = len(weights), len(weights[0])
    integer(rows, 1, 64, "global rows")
    integer(k, 1, 4096, "global K")
    for row in weights:
        if len(row) != k:
            raise ValueError("Ragged weights")
        for value in row:
            if isinstance(value, bool) or not hasattr(value, "__index__"):
                raise ValueError("Weights must be integer values")
        if materialize:
            pack(row, input_bits)
    tile_rows = min(rows, 16) if tile_rows is None else tile_rows
    tile_k = min(k, 64) if tile_k is None else tile_k
    integer(tile_rows, 1, min(rows, 16), "tile_rows")
    integer(tile_k, 1, min(k, 64), "tile_k")
    addresses = layout(tile_rows, tile_k, k, input_bits, output_bits, profile)
    commands, load_specs, stores, weights_blob, chunks = [], [], [], bytearray(), {}
    for row in range(0, rows, tile_rows):
        count = min(tile_rows, rows-row)
        for col in range(0, k, tile_k):
            size = min(tile_k, k-col)
            length = count*math.ceil(size*input_bits/8)
            chunks[(row, col)] = (len(weights_blob), length, count, size)
            if materialize:
                weights_blob.extend(b"".join(pack(w[col:col+size], input_bits) for w in weights[row:row+count]))
    pairs = ([(frame, row) for frame in range(frames) for row in range(0, rows, tile_rows)]
             if order == "frame_major" else
             [(frame, row) for row in range(0, rows, tile_rows) for frame in range(frames)])
    resident_w, resident_a = None, None
    for frame, row in pairs:
        for col in range(0, k, tile_k):
            offset, length, count, size = chunks[(row, col)]
            if not reuse_weights or resident_w != (row, col):
                load_specs.append({"command": len(commands), "source": "weights", "offset": offset, "count": length})
                commands.append({"op": "LOAD", "addr": addresses["w"], "count": length})
                resident_w = (row, col)
            # A has an invariant disjoint region even for tail row/column tiles.
            if resident_a != (frame, col, size):
                load_specs.append({"command": len(commands), "source": "input", "frame": frame,
                                   "offset": col, "elements": size})
                commands.append({"op": "LOAD", "addr": addresses["a"], "count": math.ceil(size*input_bits/8)})
                resident_a = (frame, col, size)
            commands.append({"op": "MVM", "a": addresses["a"], "w": addresses["w"], "y": addresses["y"],
                             "rows": count, "k": size, "stride": math.ceil(size*input_bits/8),
                             "input_bits": input_bits, "accumulate": col > 0,
                             "sat16": output_bits == 16 and col+size == k})
        stores.append({"command": len(commands), "frame": frame, "row": row, "rows": count, "bits": output_bits})
        commands.append({"op": "STORE", "addr": addresses["y"], "count": count*output_bits//8})
    commands.append({"op": "END"})
    binary = encode(commands, encoding)
    return {"name": name, "profile": asdict(profile), "input_bits": input_bits,
            "output_bits": output_bits, "encoding": encoding, "frames": frames,
            "rows": rows, "k": k, "tile_rows": tile_rows, "tile_k": tile_k,
            "order": order, "reuse_weights": reuse_weights, "layout": addresses,
            "commands": commands, "load_specs": load_specs, "stores": stores,
            "weights_hex": weights_blob.hex(), "commands_hex": binary.hex(),
            "predicted": predicted(commands, profile, encoding),
            "commands_sha256": hashlib.sha256(binary).hexdigest(),
            "materialized": materialize,
            "status": "experimental device ISA; no RTL/PPA evidence"}


def bind(plan, inputs):
    if not plan.get("materialized", True):
        raise ValueError("Shape estimate is not executable")
    inputs = [list(row) for row in inputs]
    if len(inputs) != plan["frames"] or any(len(row) != plan["k"] for row in inputs):
        raise ValueError("Runtime input shape mismatch")
    for row in inputs:
        if any(isinstance(v, bool) or not hasattr(v, "__index__") for v in row):
            raise ValueError("Inputs must be integer values")
        pack(row, plan["input_bits"])
    weights = bytes.fromhex(plan["weights_hex"])
    payload = bytearray()
    for spec in plan["load_specs"]:
        if spec["source"] == "weights":
            payload.extend(weights[spec["offset"]:spec["offset"]+spec["count"]])
        else:
            row = inputs[spec["frame"]]
            payload.extend(pack(row[spec["offset"]:spec["offset"]+spec["elements"]], plan["input_bits"]))
    if len(payload) != plan["predicted"]["load_bytes"]:
        raise AssertionError("Binding byte count mismatch")
    return bytes.fromhex(plan["commands_hex"]), bytes(payload)


def unpack_stores(plan, report):
    data = bytes.fromhex(report["stores_hex"])
    result = [[None]*plan["rows"] for _ in range(plan["frames"])]
    cursor = 0
    for spec in plan["stores"]:
        unit = spec["bits"]//8
        for index in range(spec["rows"]):
            result[spec["frame"]][spec["row"]+index] = int.from_bytes(data[cursor:cursor+unit], "little", signed=True)
            cursor += unit
    if cursor != len(data) or any(v is None for row in result for v in row):
        raise AssertionError("Invalid store coverage/length")
    return result


def oracle(weights, inputs, acc_bits, output_bits):
    """Independent mathematical integer loop; wraps each MAC and clamps only at end."""
    result, overflows, saturations = [], 0, 0
    low, high, modulus = -(1 << (acc_bits-1)), (1 << (acc_bits-1))-1, 1 << acc_bits
    for vector in inputs:
        outputs = []
        for row in weights:
            value = 0
            for a, w in zip(vector, row):
                before = value+int(a)*int(w)
                overflows += int(before < low or before > high)
                value = (before-low) % modulus+low
            if output_bits == 16:
                clipped = min(32767, max(-32768, value))
                saturations += int(clipped != value)
                value = clipped
            outputs.append(value)
        result.append(outputs)
    return result, overflows, saturations


class IsaRuntime:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.tlpu_isa_run.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                             ctypes.c_size_t]+[ctypes.c_uint]*7
        self.library.tlpu_isa_run.restype = ctypes.c_char_p
        self.library.tlpu_isa_last_error.argtypes = []
        self.library.tlpu_isa_last_error.restype = ctypes.c_char_p

    def run_raw(self, commands, loads, profile=None, encoding=4):
        profile = profile or Profile()
        raw = self.library.tlpu_isa_run(commands, len(commands), loads, len(loads),
                                      profile.scratch_bytes, profile.lanes, profile.depth,
                                      profile.accumulator_bits, profile.memory_byte_bw,
                                      encoding, int(profile.enable_int8))
        if raw is None:
            message = self.library.tlpu_isa_last_error()
            raise ValueError(message.decode() if message else "ISA runtime error")
        return json.loads(raw)

    def run(self, plan, inputs):
        commands, loads = bind(plan, inputs)
        report = self.run_raw(commands, loads, Profile(**plan["profile"]), plan["encoding"])
        for field, value in plan["predicted"].items():
            if report[field] != value:
                raise AssertionError(f"{field}: C++={report[field]} Python={value}")
        return unpack_stores(plan, report), report


def save_plan(plan, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"manifest.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    (directory/"commands.bin").write_bytes(bytes.fromhex(plan["commands_hex"]))
    (directory/"weights.bin").write_bytes(bytes.fromhex(plan["weights_hex"]))
