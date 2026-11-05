"""Sequential end-to-end estimates from the commands the compiler actually emits.

All transport and host costs are scenario inputs, not measured silicon properties.
Simulator/VM wall time is deliberately outside this model.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PerformanceScenario:
    name: str = "hypothetical-100MHz-10MBps"
    core_hz: float = 100_000_000
    tx_bytes_per_second: float = 10_000_000
    rx_bytes_per_second: float = 10_000_000
    device_command_bytes: int = 8
    device_setup_cycles: int = 2
    host_dispatch_ns: float = 100
    vm_offload_call_ns: float = 500
    host_quant_ns_per_element: float = 20
    host_pack_ns_per_element: float = 5
    host_unpack_ns_per_element: float = 5

    def validate(self):
        for name in ("core_hz", "tx_bytes_per_second", "rx_bytes_per_second"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        for name in ("device_command_bytes", "device_setup_cycles"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        for name in ("host_dispatch_ns", "vm_offload_call_ns", "host_quant_ns_per_element",
                     "host_pack_ns_per_element", "host_unpack_ns_per_element"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be nonnegative and finite")


def estimate_sequence(segments, scenario=None):
    """Estimate device_plan / host_relu_q34 segments in execution order.

Each device plan incurs one VM boundary. Its commands run sequentially with
no DMA/compute overlap and reload their explicitly requested weights on every
invocation. AOT compile/load costs and actual Python/C++ simulation are excluded.
"""
    scenario = scenario or PerformanceScenario()
    scenario.validate()
    components = {name: 0.0 for name in ("vm_boundary", "host_dispatch", "host_pack",
                  "host_unpack", "host_quant", "command_tx", "payload_tx", "payload_rx",
                  "device_setup", "core")}
    counts = {"core_cycles": 0, "macs": 0, "load_a_bytes": 0, "load_w_bytes": 0,
              "store_y_bytes": 0, "command_bytes": 0, "device_commands": 0,
              "host_quant_elements": 0, "vm_offload_calls": 0}
    events = []
    time_ns = 0.0

    def event(category, duration_ns, segment, instruction=None, **details):
        nonlocal time_ns
        events.append({"category": category, "start_ns": time_ns, "duration_ns": duration_ns,
                       "end_ns": time_ns + duration_ns, "segment": segment,
                       "instruction": instruction, **details})
        time_ns += duration_ns
        components[category] += duration_ns

    for index, segment in enumerate(segments):
        if segment["kind"] == "host_relu_q34":
            count = segment["elements"]
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid host quantization element count")
            event("host_quant", count * scenario.host_quant_ns_per_element, index)
            counts["host_quant_elements"] += count
            continue
        if segment["kind"] != "device_plan":
            raise ValueError("Unknown execution segment")
        manifest = segment["manifest"]
        event("vm_boundary", scenario.vm_offload_call_ns, index)
        counts["vm_offload_calls"] += 1
        lanes, depth = manifest["target"]["lanes"], manifest["target"]["depth"]
        observed = {"load_a_bytes": 0, "load_w_bytes": 0, "store_y_bytes": 0,
                    "core_cycles": 0, "macs": 0, "host_quant_elements": 0}
        for command_index, instruction in enumerate(manifest["instructions"]):
            op, args = instruction["op"], instruction["args"]
            event("host_dispatch", scenario.host_dispatch_ns, index, command_index, opcode=op)
            if op in ("LOAD_A", "LOAD_W", "GEMV", "STORE_Y"):
                event("command_tx", scenario.device_command_bytes * 1e9 / scenario.tx_bytes_per_second,
                      index, command_index, opcode=op, bytes=scenario.device_command_bytes)
                event("device_setup", scenario.device_setup_cycles * 1e9 / scenario.core_hz,
                      index, command_index, opcode=op)
                counts["command_bytes"] += scenario.device_command_bytes
                counts["device_commands"] += 1
            if op in ("LOAD_A", "LOAD_W"):
                payload = (args[2]+1)//2 if op == "LOAD_A" else args[1]
                if op == "LOAD_A":
                    event("host_pack", args[2] * scenario.host_pack_ns_per_element, index, command_index)
                event("payload_tx", payload * 1e9 / scenario.tx_bytes_per_second,
                      index, command_index, opcode=op, bytes=payload)
                observed["load_a_bytes" if op == "LOAD_A" else "load_w_bytes"] += payload
            elif op == "STORE_Y":
                payload = args[3] * 2
                event("payload_rx", payload * 1e9 / scenario.rx_bytes_per_second,
                      index, command_index, opcode=op, bytes=payload)
                event("host_unpack", args[3] * scenario.host_unpack_ns_per_element, index, command_index)
                observed["store_y_bytes"] += payload
            elif op == "GEMV":
                rows, k = args[3:5]
                cycles = math.ceil(rows/lanes) * k + depth - 1
                event("core", cycles * 1e9 / scenario.core_hz, index, command_index, core_cycles=cycles)
                observed["core_cycles"] += cycles
                observed["macs"] += rows * k
            elif op == "HOST_RELU_Q":
                event("host_quant", args[2] * scenario.host_quant_ns_per_element, index, command_index)
                observed["host_quant_elements"] += args[2]
            elif op not in ("ALLOC_TENSOR", "END"):
                raise ValueError("Unknown plan opcode " + op)
        for name, value in observed.items():
            counts[name] += value
        expected = {**manifest["predicted_transfers"], "core_cycles": manifest["predicted_core_cycles"],
                    "macs": manifest["predicted_macs"], "host_quant_elements": manifest["predicted_host_ops"]}
        if observed != expected:
            raise ValueError("Manifest prediction differs from command accounting")
    if not counts["vm_offload_calls"]:
        raise ValueError("At least one device plan is required")
    total = sum(components.values())
    return {"scenario": asdict(scenario), "counts": counts, "components_ns": components,
            "total_ns": total, "core_fraction": components["core"]/total,
            "inferences_per_second": 1e9/total, "effective_macs_per_second": counts["macs"]*1e9/total,
            "events": events, "policy": "synchronous sequential commands; no overlap; weights reloaded per plan invocation",
            "physical_performance_validated": False,
            "excluded": ["AOT compile and artifact load", "simulator/VM measured wall time",
                         "unconfigured board/driver stalls, protocol ACKs and framing beyond device_command_bytes"]}
