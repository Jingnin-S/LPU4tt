"""Lower the existing real Relax frontend's Tiny IR to the new device stream.

The legacy Tiny IR declares INT16 GEMV outputs. Reject shapes/weights that could
overflow that type; do not silently reinterpret it as a 32-bit tensor.
"""

from .isa import Profile, compile_linear, layout, save_plan, time_seconds


class IsaGraph:
    def __init__(self, output_node, directory, frames=1, profile=None, encoding=4):
        self.profile = profile or Profile()
        if self.profile.accumulator_bits != 32:
            raise ValueError("Graph adapter requires the ACC32 candidate")
        self.frames, self.nodes, self.plans = frames, [], {}

        def visit(node):
            if node in self.nodes:
                return
            if node.source is not None:
                visit(node.source)
            self.nodes.append(node)

        visit(output_node)
        inputs = [n for n in self.nodes if n.op == "input"]
        if len(inputs) != 1:
            raise ValueError("Graph requires exactly one input")
        self.input, self.output = inputs[0], output_node
        for node in self.nodes:
            if node.op == "gemv":
                for row in node.weights:
                    upper = sum(max(-8*w, 7*w) for w in row)
                    lower = sum(min(-8*w, 7*w) for w in row)
                    if lower < -32768 or upper > 32767:
                        raise ValueError("Legacy Tiny IR INT16 output is not provably safe; explicit INT32 IR needed")
                candidates = []
                for tm in sorted({1, 2, 4, 8, min(node.size, 16)}):
                    if tm > min(node.size, 16):
                        continue
                    for tk in sorted({1, 8, 16, 32, min(node.source.size, 64)}):
                        if tk > min(node.source.size, 64):
                            continue
                        try:
                            layout(tm, tk, node.source.size, 4, 16, self.profile)
                        except ValueError:
                            continue
                        plan = compile_linear(node.weights, frames, self.profile, 4, 16, encoding, tm, tk)
                        candidates.append((time_seconds(plan["predicted"]), tm, tk, plan))
                if not candidates:
                    raise ValueError("No legal graph tile")
                plan = min(candidates, key=lambda item: item[:3])[-1]
                self.plans[node] = plan
                save_plan(plan, directory/f"partition_{len(self.plans)-1:02d}")
            elif node.op == "host_relu_q34":
                if node.source.bits != 16 or node.bits != 4:
                    raise ValueError("Invalid host quantization type")
            elif node.op != "input":
                raise ValueError("Unsupported graph operator")

    def run(self, runtime, inputs):
        values, reports, host_elements = {self.input: inputs}, [], 0
        for node in self.nodes:
            if node.op == "gemv":
                values[node], report = runtime.run(self.plans[node], values[node.source])
                reports.append(report)
            elif node.op == "host_relu_q34":
                values[node] = [[min(7, (max(0, value)*3+2)//4) for value in row]
                                for row in values[node.source]]
                host_elements += sum(map(len, values[node]))
        return values[self.output], reports, host_elements
