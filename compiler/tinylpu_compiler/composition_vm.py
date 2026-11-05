"""Actual Relax VM backend for ISA v0.1 and compiler-visible LPU composition."""
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from .architecture import ArchRuntime, Resources, choose_plan
from .isa import IsaRuntime, Profile, decode, save_plan


class CompositionVm:
    def __init__(self, outputs, directory, isa_library, architecture_library,
                 frames=1, profile=None, resources=None, storage="bank4", fuse_qkv=False):
        import numpy as np
        import tvm
        from tvm import relax
        self.owner = threading.get_ident()
        self.closed, self.callback_names, self.callbacks = False, [], []
        self.last_reports, self.partitions, self.last_timing = [], [], {}
        self.total_callback_calls = 0
        self.host_quant_elements = 0
        self.folder = Path(directory)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        self.profile, self.resources = profile or Profile(memory_byte_bw=1), resources or Resources()
        self.runtime, self.analyzer = IsaRuntime(isa_library), ArchRuntime(architecture_library)
        self.frames, self.nodes = frames, []

        def visit(node):
            if node in self.nodes:
                return
            if node.source is not None:
                visit(node.source)
            self.nodes.append(node)
        for node in self.outputs:
            visit(node)
        inputs = [node for node in self.nodes if node.op == "input"]
        if len(inputs) != 1 or not 1 <= frames <= 32:
            raise ValueError("Composition requires one input and batch 1..32")
        self.input_elements = inputs[0].size
        if self.profile.accumulator_bits != 32:
            raise ValueError("New VM backend requires ACC32")
        for node in self.nodes:
            if node.op == "gemv":
                for row in node.weights:
                    if sum(max(-8*w, 7*w) for w in row) > 32767 or sum(min(-8*w, 7*w) for w in row) < -32768:
                        raise ValueError("INT16 frontend output range cannot be proven")
            elif node.op not in ("input", "host_relu_q34"):
                raise ValueError("Unsupported composition op")
        fusion = self.outputs if fuse_qkv and len(self.outputs) > 1 else ()
        if fusion and (any(n.op != "gemv" for n in fusion) or
                       any(n.source is not fusion[0].source for n in fusion) or
                       sum(n.size for n in fusion) > 64):
            raise ValueError("Fusion requires compatible sibling GEMVs with one common source, <=64 rows")
        builder = relax.BlockBuilder()
        x = relax.Var("input", relax.TensorType((frames, self.input_elements), "float32"))
        expressions = {inputs[0]: x}
        prefix, build_start = "tinylpu.v01."+uuid.uuid4().hex, time.perf_counter_ns()

        def partition(members):
            weights = [list(row) for member in members for row in member.weights]
            plan, resource_report, candidates = choose_plan(weights, frames, self.profile, self.resources,
                                                           self.analyzer, storage, output_bits=16)
            index = len(self.partitions)
            folder = self.folder/f"partition_{index:02d}"
            save_plan(plan, folder)
            mapping = {"partition": index, "nodes": [self.nodes.index(n) for n in members],
                       "source_node": self.nodes.index(members[0].source),
                       "plan": plan, "resources": resource_report, "search": candidates,
                       "placement": "device ISA v0.1; software C++ simulator"}
            self.partitions.append(mapping)
            (folder/"debug.json").write_text(json.dumps({**mapping, "disassembly": decode(bytes.fromhex(plan["commands_hex"]))}, indent=2)+"\n", encoding="utf-8")

            def callback(source, destination):
                data = source.numpy()
                if not np.all(np.isfinite(data)) or not np.all(data == np.rint(data)) or np.min(data) < -8 or np.max(data) > 7:
                    raise ValueError("Callback input left declared INT4 carrier")
                start = time.perf_counter_ns()
                result, report = self.runtime.run(plan, data.astype(np.int32).tolist())
                output = np.asarray(result, dtype=np.float32)
                destination.copyfrom(output)
                self.total_callback_calls += 1
                self.last_reports.append({"partition": index, "nodes": mapping["nodes"], "runtime": report,
                                          "resources": resource_report, "cpp_wall_ns": time.perf_counter_ns()-start,
                                          "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                                          "output_sha256": hashlib.sha256(output.tobytes()).hexdigest()})
            name = prefix+f".p{index}"
            tvm.register_global_func(name, callback, override=False)
            self.callbacks.append(callback)
            self.callback_names.append(name)
            value = builder.emit(relax.call_dps_packed(name, (expressions[members[0].source],),
                                  out_ty=relax.TensorType((frames, len(weights)), "float32")))
            start = 0
            for member in members:
                expressions[member] = (value if len(members) == 1 else builder.emit(
                    relax.op.strided_slice(value, axes=[1], begin=[start], end=[start+member.size])))
                start += member.size

        try:
            with builder.function("main", [x]):
                with builder.dataflow():
                    for node in self.nodes:
                        if node in expressions:
                            continue
                        if node.op == "gemv":
                            partition(fusion if node in fusion else (node,))
                        else:
                            source = builder.emit(relax.op.nn.relu(expressions[node.source]))
                            source = builder.emit(relax.op.multiply(source, relax.const(.75, "float32")))
                            source = builder.emit(relax.op.add(source, relax.const(.5, "float32")))
                            source = builder.emit(relax.op.floor(source))
                            expressions[node] = builder.emit(relax.op.clip(source, 0, 7))
                            self.host_quant_elements += frames*node.size
                    value = (expressions[self.outputs[0]] if len(self.outputs) == 1 else
                             relax.Tuple([expressions[n] for n in self.outputs]))
                    result = builder.emit_output(value)
                builder.emit_func_output(result)
            module = builder.get()
            (self.folder/"vm_relax.ir.txt").write_text(module.script(), encoding="utf-8")
            self.executable = relax.build(module, target="llvm")
            self.vm = relax.VirtualMachine(self.executable, tvm.cpu())
            self.build_wall_ns = time.perf_counter_ns()-build_start
            graph = [{"id": i, "op": n.op, "size": n.size, "bits": n.bits,
                      "source": self.nodes.index(n.source) if n.source is not None else None,
                      "placement": "host" if n.op == "host_relu_q34" else "input" if n.op == "input" else "device"}
                     for i, n in enumerate(self.nodes)]
            (self.folder/"graph_debug.json").write_text(json.dumps({"nodes": graph, "fused_qkv": bool(fusion),
                 "frames": frames, "host_quant_elements": self.host_quant_elements,
                 "partitions": [{k:v for k,v in p.items() if k not in ("plan", "search")} for p in self.partitions]}, indent=2)+"\n", encoding="utf-8")
        except Exception:
            self.close()
            raise

    def run(self, value):
        import numpy as np
        import tvm
        self._check_thread()
        self.last_reports, self.last_timing = [], {}
        if self.closed:
            raise RuntimeError("VM is closed")
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or
            value.shape != (self.frames, self.input_elements) or not np.all(np.isfinite(value)) or
            not np.all(value == np.rint(value)) or np.min(value) < -8 or np.max(value) > 7):
            raise ValueError("Expected finite exact INT4 float32 carrier [frames,K]")
        start = time.perf_counter_ns()
        try:
            result = self.vm["main"](tvm.runtime.tensor(value))
            arrays = [result.numpy()] if len(self.outputs) == 1 else [item.numpy() for item in result]
            if len(self.last_reports) != len(self.partitions):
                raise RuntimeError("VM did not execute the expected partitions")
        except Exception:
            self.last_reports = []
            raise
        self.last_timing = {"vm_wall_ns": time.perf_counter_ns()-start,
                            "cpp_wall_ns": sum(r["cpp_wall_ns"] for r in self.last_reports)}
        return arrays[0] if len(arrays) == 1 else tuple(arrays)

    def _check_thread(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("VM belongs to its creating thread")

    def close(self):
        self._check_thread()
        if not self.closed:
            self.closed = True
            import tvm_ffi
            for name in self.callback_names:
                tvm_ffi.remove_global_func(name)
            self.callback_names.clear()
            self.callbacks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
