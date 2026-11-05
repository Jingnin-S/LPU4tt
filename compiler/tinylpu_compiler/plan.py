import hashlib
import json
from pathlib import Path
import struct

from .target import Target

MAGIC = b"TLPUv001"
HEADER = struct.Struct("<8s12I")
INSTRUCTION = struct.Struct("<8I")
OPCODES = {"ALLOC_TENSOR": 1, "LOAD_A": 2, "LOAD_W": 3, "GEMV": 4,
           "STORE_Y": 5, "HOST_RELU_Q": 6, "END": 7}


def pack_row(values):
    encoded = bytearray((len(values) + 1) // 2)
    for i, value in enumerate(values):
        if type(value) is not int or not -8 <= value <= 7:
            raise ValueError("Packing requires signed INT4")
        encoded[i // 2] |= (value & 15) << ((i % 2) * 4)
    return bytes(encoded)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def compile_plan(output_node, output_dir, target=None, name="model"):
    target = target or Target()
    target.validate()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    instructions, weights, tensors, kernels = [], bytearray(), {}, []
    input_node = None
    a_resident = None
    next_id = 1
    predicted_core_cycles = 0
    transfer = {"load_a_bytes": 0, "load_w_bytes": 0, "store_y_bytes": 0}
    host_ops = 0

    def emit(op, *args):
        if len(args) > 7 or any(type(v) is not int or not 0 <= v <= 0xffffffff for v in args):
            raise ValueError("Invalid instruction operand")
        instructions.append([OPCODES[op], *args, *([0] * (7-len(args)))])

    def visit(node):
        nonlocal input_node, next_id, a_resident, predicted_core_cycles, host_ops
        if node in tensors:
            return tensors[node]
        if node.op == "input":
            if input_node is not None and input_node is not node:
                raise ValueError("ABI v1 supports exactly one runtime input")
            input_node = node
            tensors[node] = 0
            return 0
        if node.source is None:
            raise ValueError("Missing source dependency")
        src = visit(node.source)
        dst = next_id
        next_id += 1
        if next_id > 64:
            raise ValueError("Host tensor count exceeds ABI")
        emit("ALLOC_TENSOR", dst, node.size, node.bits)
        tensors[node] = dst
        if node.op == "gemv":
            k = node.source.size
            rows_per_tile = target.tile_rows(node.size, k)
            for start in range(0, node.size, rows_per_tile):
                tile = node.weights[start:start+rows_per_tile]
                rows = len(tile)
                layout = target.layout(rows, k)
                slot = layout["slots"][0]
                packed = b"".join(pack_row(row) for row in tile)
                offset = len(weights)
                weights.extend(packed)
                emit("LOAD_W", offset, len(packed), layout["w_addr"])
                transfer["load_w_bytes"] += len(packed)
                if a_resident is not None:
                    _, _, count, old_addr = a_resident
                    if layout["w_addr"] < old_addr + (count+1)//2 and old_addr < layout["w_addr"] + len(packed):
                        a_resident = None
                wanted = (src, 0, k, slot["a_addr"])
                if a_resident != wanted:
                    emit("LOAD_A", src, 0, k, slot["a_addr"])
                    transfer["load_a_bytes"] += (k+1)//2
                    a_resident = wanted
                emit("GEMV", slot["a_addr"], layout["w_addr"], slot["y_addr"], rows, k)
                emit("STORE_Y", slot["y_addr"], dst, start, rows)
                transfer["store_y_bytes"] += rows * 2
                cycles = ((rows + target.lanes - 1)//target.lanes)*k+target.depth-1
                predicted_core_cycles += cycles
                kernels.append({"output_tensor": dst, "output_offset": start, "rows": rows, "k": k,
                                "layout": layout, "predicted_core_cycles": cycles})
        elif node.op == "host_relu_q34":
            emit("HOST_RELU_Q", src, dst, node.size, 3, 2)
            host_ops += node.size
        else:
            raise ValueError("Unsupported target/host operator " + node.op)
        return dst

    output_id = visit(output_node)
    if input_node is None or output_node.bits != 16:
        raise ValueError("Experiment expects one INT4 input and INT16 graph output")
    emit("END", output_id)
    code = HEADER.pack(MAGIC, 1, target.lanes, target.depth, target.a_ports, target.b_ports,
                       target.scratchpad_bytes, next_id, input_node.size, output_id, output_node.size,
                       len(instructions), len(weights))
    code += b"".join(INSTRUCTION.pack(*instruction) for instruction in instructions)
    (folder / "program.tlp").write_bytes(code)
    (folder / "weights.bin").write_bytes(weights)
    inverse = {v:k for k,v in OPCODES.items()}
    manifest = {"name": name, "abi": 1, "target": target.to_dict(),
                "input_contract": {"elements": input_node.size, "dtype": "signed_int4", "range": [-8,7]},
                "output_contract": {"elements": output_node.size, "dtype": "signed_int16"},
                "program_sha256": sha256(code), "weights_sha256": sha256(weights),
                "program_bytes": len(code), "weights_bytes": len(weights), "host_tensor_count": next_id,
                "instructions": [{"op":inverse[i[0]], "args":i[1:]} for i in instructions],
                "kernels": kernels, "predicted_core_cycles": predicted_core_cycles,
                "predicted_macs": sum(k["rows"]*k["k"] for k in kernels),
                "predicted_transfers": transfer, "predicted_host_ops": host_ops,
                "execution_policy": "sequential host command plan; buffers reserved, no overlap claimed",
                "isa_status": "experimental host/device command ABI, not frozen RTL ISA",
                "physical_performance_validated": False}
    tiny_ir = {"input_tensor": 0, "output_tensor": output_id,
               "nodes": [{"tensor":tensor_id, "op":node.op, "elements":node.size,
                          "bits":node.bits, "source":tensors[node.source] if node.source is not None else None,
                          "weight_shape":[len(node.weights),len(node.weights[0])] if node.weights else None}
                         for node,tensor_id in sorted(tensors.items(),key=lambda item:item[1])]}
    (folder / "tiny_ir.json").write_text(json.dumps(tiny_ir, indent=2)+"\n", encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def verify_artifact(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for filename, key in (("program.tlp", "program_sha256"), ("weights.bin", "weights_sha256")):
        if sha256((folder / filename).read_bytes()) != manifest[key]:
            raise ValueError("Artifact hash mismatch: " + filename)
    return manifest
