"""ABI-v1 programs for a cold batch of independent INT4 linear requests."""

import json
from pathlib import Path

from .ir import gemv, input_vector
from .plan import HEADER, INSTRUCTION, MAGIC, OPCODES, pack_row, sha256
from .target import Target


def compile_batch_linear(weights, frames, output_dir, target=None, name="linear",
                         order="latency_first", weight_policy="reload"):
    target = target or Target()
    target.validate()
    weights = tuple(tuple(row) for row in weights)
    if not weights or not weights[0]:
        raise ValueError("Nonempty INT4 weights required")
    k, rows = len(weights[0]), len(weights)
    gemv(input_vector(k), weights)  # reuse the declared integer/shape contract
    if type(frames) is not int or frames < 1 or max(frames*k, frames*rows) > 4096:
        raise ValueError("Batch input/output must contain 1..4096 elements")
    if order not in ("latency_first", "weight_stationary"):
        raise ValueError("Unknown batch execution order")
    if weight_policy not in ("reload", "resident"):
        raise ValueError("Unknown weight policy")
    size = target.tile_rows(rows, k)
    instructions, kernels, jobs, tiles, blob = [], [], [], [], bytearray()
    transfers = {"load_a_bytes": 0, "load_w_bytes": 0, "store_y_bytes": 0}

    def emit(op, *args):
        index = len(instructions)
        instructions.append({"op": op, "args": [*args, *([0]*(7-len(args)))]})
        return index

    for offset in range(0, rows, size):
        tile_weights = weights[offset:offset+size]
        payload = b"".join(pack_row(row) for row in tile_weights)
        tiles.append({"tile": len(tiles), "output_offset": offset, "rows": len(tile_weights),
                      "blob_offset": len(blob), "weight_bytes": len(payload),
                      "layout": target.layout(len(tile_weights), k)})
        blob.extend(payload)
    emit("ALLOC_TENSOR", 1, frames*rows, 16)
    pairs = ([(frame, tile) for frame in range(frames) for tile in tiles]
             if order == "latency_first" else
             [(frame, tile) for tile in tiles for frame in range(frames)])
    resident_weight = None
    resident_a = []  # (frame, k, byte address), invalidated by any overlapping write

    def invalidate(address, count):
        resident_a[:] = [item for item in resident_a
                         if not (address < item[2]+(item[1]+1)//2 and item[2] < address+count)]

    for frame, tile in pairs:
        layout = tile["layout"]
        slot_index = frame % target.buffers
        slot = layout["slots"][slot_index]
        indices = {"load_w": None, "load_a": None}
        if weight_policy == "reload" or resident_weight != tile["tile"]:
            indices["load_w"] = emit("LOAD_W", tile["blob_offset"], tile["weight_bytes"], layout["w_addr"])
            transfers["load_w_bytes"] += tile["weight_bytes"]
            invalidate(layout["w_addr"], tile["weight_bytes"])
            resident_weight = tile["tile"]
        wanted = (frame, k, slot["a_addr"])
        if wanted not in resident_a:
            indices["load_a"] = emit("LOAD_A", 0, frame*k, k, slot["a_addr"])
            transfers["load_a_bytes"] += (k+1)//2
            invalidate(slot["a_addr"], (k+1)//2)
            resident_a.append(wanted)
        count = tile["rows"]
        indices["gemv"] = emit("GEMV", slot["a_addr"], layout["w_addr"], slot["y_addr"], count, k)
        invalidate(slot["y_addr"], 2*count)
        indices["store_y"] = emit("STORE_Y", slot["y_addr"], 1, frame*rows+tile["output_offset"], count)
        transfers["store_y_bytes"] += 2*count
        cycles = ((count+target.lanes-1)//target.lanes)*k+target.depth-1
        kernels.append({"output_tensor": 1, "output_offset": frame*rows+tile["output_offset"],
                        "rows": count, "k": k, "layout": layout, "predicted_core_cycles": cycles})
        jobs.append({"frame": frame, "tile": tile["tile"], "rows": count, "k": k, "slot": slot_index,
                     "a_addr": slot["a_addr"], "w_addr": layout["w_addr"], "y_addr": slot["y_addr"],
                     "instruction_indices": indices})
    emit("END", 1)
    if len(instructions) > 100000:
        raise ValueError("Too many ABI-v1 instructions")
    program = HEADER.pack(MAGIC, 1, target.lanes, target.depth, target.a_ports, target.b_ports,
                          target.scratchpad_bytes, 2, frames*k, 1, frames*rows, len(instructions), len(blob))
    program += b"".join(INSTRUCTION.pack(OPCODES[item["op"]], *item["args"]) for item in instructions)
    manifest = {"name": name, "abi": 1, "target": target.to_dict(), "frames": frames,
                "output_rows": rows, "k": k, "order": order, "weight_policy": weight_policy,
                "input_contract": {"elements": frames*k, "dtype": "signed_int4", "range": [-8,7],
                                   "layout": "frame-major; runtime carrier shape [1,frames*K]"},
                "output_contract": {"elements": frames*rows, "dtype": "signed_int16", "layout": "frame-major"},
                "program_sha256": sha256(program), "weights_sha256": sha256(blob),
                "program_bytes": len(program), "weights_bytes": len(blob), "host_tensor_count": 2,
                "instructions": instructions, "kernels": kernels, "jobs": jobs, "tiles": tiles,
                "predicted_core_cycles": sum(kernel["predicted_core_cycles"] for kernel in kernels),
                "predicted_macs": frames*rows*k, "predicted_transfers": transfers, "predicted_host_ops": 0,
                "execution_policy": "cold batch; ABI-v1 sequential commands; single mutable W region; frame%buffers A/Y slot",
                "isa_status": "experimental host/device command ABI, not frozen RTL ISA",
                "physical_performance_validated": False}
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"program.tlp").write_bytes(program)
    (folder/"weights.bin").write_bytes(blob)
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest
