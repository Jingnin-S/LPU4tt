"""Explicit Tiny-IR-to-Relax VM lowering for the supported integer graph subset.

The Tiny IR is imported from the real BYOC-partitioned model. This bridge is
not a generic RunCodegen plugin: GEMVs call the loaded C++ model, while the
declared quantization is compiled as native Relax CPU operations.
"""

from pathlib import Path
import threading
import time
import uuid

from .ir import gemv, input_vector
from .plan import compile_plan
from .runtime import PlanSession, validate_carrier


class VmProgram:
    def __init__(self, output_node, output_dir, library, target=None):
        import numpy as np
        import tvm
        from tvm import relax

        self._lock = threading.RLock()
        self._owner_thread = threading.get_ident()
        self._closed = False
        self.sessions = []
        self.segments = []
        self.callbacks = []
        self.last_reports = []
        self.last_timing = {}
        self.total_callback_calls = 0
        self.host_quant_elements = 0
        self._callback_ns = []
        self._cpp_ns = []
        self._callback_names = []
        self._prefix = "tinylpu.m2." + uuid.uuid4().hex
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        nodes = []

        def visit(node):
            if node in nodes:
                return
            if node.source is not None:
                visit(node.source)
            nodes.append(node)

        visit(output_node)
        inputs = [node for node in nodes if node.op == "input"]
        if len(inputs) != 1:
            raise ValueError("VM bridge requires exactly one graph input")
        self.input_elements = inputs[0].size
        self.output_elements = output_node.size
        build_start = time.perf_counter_ns()
        builder = relax.BlockBuilder()
        x = relax.Var("input", relax.TensorType((1, self.input_elements), "float32"))
        expressions = {inputs[0]: x}

        def make_callback(session, plan_index):
            def callback(source, destination):
                start = time.perf_counter_ns()
                report = session.run(source.numpy())
                destination.copyfrom(np.asarray(report["result"], dtype="float32").reshape(1, -1))
                self.last_reports.append({"plan_index": plan_index, **report})
                self.total_callback_calls += 1
                self._cpp_ns.append(session.last_cpp_call_ns)
                self._callback_ns.append(time.perf_counter_ns()-start)
            return callback

        try:
            with builder.function("main", [x]):
                with builder.dataflow():
                    for node in nodes:
                        if node.op == "input":
                            continue
                        source = expressions[node.source]
                        if node.op == "gemv":
                            index = len(self.sessions)
                            folder = self.folder / ("partition_{:02d}".format(index))
                            manifest = compile_plan(gemv(input_vector(node.source.size), node.weights),
                                                    folder, target, "partition_{}".format(index))
                            session = PlanSession(folder, library)
                            self.sessions.append(session)
                            self.segments.append({"kind": "device_plan", "manifest": manifest})
                            name = self._prefix + ".gemv{}".format(index)
                            callback = make_callback(session, index)
                            self.callbacks.append(callback)
                            tvm.register_global_func(name, callback, override=False)
                            self._callback_names.append(name)
                            value = relax.call_dps_packed(name, (source,),
                                            out_ty=relax.TensorType((1, node.size), "float32"))
                        elif node.op == "host_relu_q34":
                            value = builder.emit(relax.op.nn.relu(source))
                            value = builder.emit(relax.op.multiply(value, relax.const(0.75, "float32")))
                            value = builder.emit(relax.op.add(value, relax.const(0.5, "float32")))
                            value = builder.emit(relax.op.floor(value))
                            value = relax.op.clip(value, 0, 7)
                            self.segments.append({"kind": "host_relu_q34", "elements": node.size})
                            self.host_quant_elements += node.size
                        else:
                            raise ValueError("Unsupported VM graph node " + node.op)
                        expressions[node] = builder.emit(value)
                    result = builder.emit_output(expressions[output_node])
                builder.emit_func_output(result)
            module = builder.get()
            (self.folder / "vm_relax.ir.txt").write_text(module.script(), encoding="utf-8")
            self.executable = relax.build(module, target="llvm")
            self.vm = relax.VirtualMachine(self.executable, tvm.cpu())
            self.build_wall_ns = time.perf_counter_ns()-build_start
            self.artifact_load_wall_ns = sum(session.load_wall_ns for session in self.sessions)
        except Exception:
            self.close()
            raise

    def run(self, value):
        import tvm

        self._check_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("VM program is closed")
            self.last_reports = []
            self.last_timing = {}
            self._callback_ns = []
            self._cpp_ns = []
            validate_carrier(value, self.input_elements)
            start = time.perf_counter_ns()
            tensor = tvm.runtime.tensor(value)
            execute_start = time.perf_counter_ns()
            try:
                result = self.vm["main"](tensor)
                vm_ns = time.perf_counter_ns()-execute_start
                output = result.numpy()
                total_ns = time.perf_counter_ns()-start
                if len(self.last_reports) != len(self.sessions):
                    raise RuntimeError("VM did not execute every expected device partition")
            except Exception:
                # Partial device outputs must not masquerade as a completed
                # inference; the next call still starts from clean state.
                self.last_reports = []
                self._callback_ns = []
                self._cpp_ns = []
                raise
            self.last_timing = {"wrapper_wall_ns": total_ns, "vm_execute_ns": vm_ns,
                                "offload_callbacks_ns": sum(self._callback_ns),
                                "cpp_simulator_calls_ns": sum(self._cpp_ns),
                                "vm_outside_callbacks_ns": vm_ns-sum(self._callback_ns)}
            return output

    def close(self):
        self._check_thread()
        with self._lock:
            if not self._closed:
                self._closed = True
                for session in self.sessions:
                    session.close()
                # Remove registered Python closures so repeated builds do not
                # keep closed C++ sessions alive in the process-wide registry.
                import tvm_ffi
                for name in self._callback_names:
                    tvm_ffi.remove_global_func(name)
                self._callback_names.clear()

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("VM program must be used from its creating thread")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
