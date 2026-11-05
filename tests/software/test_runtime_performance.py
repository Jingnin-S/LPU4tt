from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compiler"))
from tinylpu_compiler import compile_plan, gemv, input_vector
from tinylpu_compiler.performance import PerformanceScenario, estimate_sequence


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as folder:
            self.manifest = compile_plan(gemv(input_vector(4), [[1]*4, [-1]*4]), folder)
        self.segments = [{"kind": "device_plan", "manifest": self.manifest}]

    def test_exact_sequential_sum_and_payloads(self):
        scenario = PerformanceScenario(core_hz=100_000_000, tx_bytes_per_second=1e9,
                                       rx_bytes_per_second=1e9, device_command_bytes=8,
                                       device_setup_cycles=2, host_dispatch_ns=0,
                                       vm_offload_call_ns=0, host_pack_ns_per_element=0,
                                       host_unpack_ns_per_element=0)
        result = estimate_sequence(self.segments, scenario)
        # 4 device commands, A2+W4+Y4 payload bytes, 6 core cycles, 8 setup cycles.
        self.assertEqual(result["total_ns"], 32+10+60+80)
        self.assertEqual(result["counts"]["command_bytes"], 32)
        self.assertEqual(result["counts"]["load_a_bytes"], 2)
        self.assertEqual(result["counts"]["store_y_bytes"], 4)
        self.assertAlmostEqual(sum(e["duration_ns"] for e in result["events"]), result["total_ns"])
        for previous, current in zip(result["events"], result["events"][1:]):
            self.assertEqual(previous["end_ns"], current["start_ns"])

    def test_bandwidth_and_clock_sensitivity_and_host_costs(self):
        base = PerformanceScenario()
        slow = estimate_sequence(self.segments, base)
        fast = estimate_sequence(self.segments, replace(base, tx_bytes_per_second=1e9, rx_bytes_per_second=1e9))
        self.assertLess(fast["total_ns"], slow["total_ns"])
        faster_clock = estimate_sequence(self.segments, replace(base, core_hz=2e8))
        self.assertEqual(faster_clock["components_ns"]["payload_tx"], slow["components_ns"]["payload_tx"])
        self.assertEqual(faster_clock["components_ns"]["core"], slow["components_ns"]["core"] / 2)
        quant = estimate_sequence(self.segments + [{"kind": "host_relu_q34", "elements": 2}], base)
        self.assertEqual(quant["total_ns"]-slow["total_ns"], 2*base.host_quant_ns_per_element)
        self.assertEqual(quant["counts"]["host_quant_elements"], 2)

    def test_invalid_scenarios_and_inconsistent_artifact_fail(self):
        for field, value in (("core_hz", 0), ("tx_bytes_per_second", float("nan")),
                             ("device_command_bytes", 1.5), ("host_dispatch_ns", -1),
                             ("core_hz", True), ("host_dispatch_ns", False)):
            with self.assertRaises(ValueError):
                estimate_sequence(self.segments, replace(PerformanceScenario(), **{field: value}))
        self.manifest["predicted_core_cycles"] += 1
        with self.assertRaisesRegex(ValueError, "Manifest"):
            estimate_sequence(self.segments)


if __name__ == "__main__":
    unittest.main()
