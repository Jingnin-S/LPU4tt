from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"compiler"))
from tinylpu_compiler import Target
from tinylpu_compiler.batch_plan import compile_batch_linear
from tinylpu_compiler.comparison import compare_schedule, validate_calendar
from tinylpu_compiler.performance import PerformanceScenario, estimate_sequence


class ComparisonTests(unittest.TestCase):
    def compile(self, rows=8, k=16, frames=8, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            return compile_batch_linear([[((r+i)%5)-2 for i in range(k)] for r in range(rows)],
                                        frames, folder, **kwargs)

    def test_weight_stationary_counts_real_a_tradeoff_and_cold_load(self):
        reload = self.compile(weight_policy="reload")
        resident = self.compile(weight_policy="resident")
        stationary = self.compile(order="weight_stationary", weight_policy="resident")
        self.assertEqual(reload["predicted_transfers"]["load_w_bytes"], 8*64)
        self.assertEqual(resident["predicted_transfers"]["load_w_bytes"], 8*64)
        self.assertEqual(stationary["predicted_transfers"]["load_w_bytes"], 64)
        self.assertEqual(resident["predicted_transfers"]["load_a_bytes"], 8*8)
        self.assertEqual(stationary["predicted_transfers"]["load_a_bytes"], 8*2*8)
        self.assertEqual(resident["predicted_core_cycles"], stationary["predicted_core_cycles"])
        one_tile = self.compile(rows=4, weight_policy="resident")
        self.assertEqual(one_tile["predicted_transfers"]["load_w_bytes"], 32)

    def test_sequential_model_matches_m2_and_frame_completion(self):
        manifest = self.compile()
        reference = estimate_sequence([{"kind":"device_plan", "manifest":manifest}])
        result = compare_schedule(manifest, capture=True)
        self.assertEqual(result["total_ns"], reference["total_ns"])
        self.assertEqual(result["counts"], reference["counts"])
        self.assertLess(result["first_frame_ns"], result["total_ns"])
        self.assertEqual(len(result["frame_completion_ns"]), 8)
        self.assertAlmostEqual(result["amortized_frame_ns"]*8, result["total_ns"])
        self.assertTrue(validate_calendar(result))

    def test_overlap_has_resource_and_address_safety_with_weight_switches(self):
        manifest = self.compile(frames=8, order="weight_stationary", weight_policy="resident")
        sequential = compare_schedule(manifest)
        overlap = compare_schedule(manifest, overlap=True, capture=True)
        self.assertLess(overlap["total_ns"], sequential["total_ns"])
        self.assertGreater(overlap["overlap_saved_ns"], 0)
        self.assertTrue(validate_calendar(overlap))
        for utilization in overlap["resource_utilization"].values():
            self.assertLessEqual(utilization, 1+1e-9)
        # Deliberately overlap two core jobs to ensure the checker is independent.
        cores = [event for event in overlap["events"] if event["resource"] == "core"]
        cores[1]["start_ns"] = cores[0]["start_ns"]
        cores[1]["end_ns"] = cores[0]["end_ns"]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_calendar(overlap)

    def test_batch_latency_and_clock_bandwidth_are_distinct(self):
        latency = self.compile(order="latency_first", weight_policy="resident")
        stationary = self.compile(order="weight_stationary", weight_policy="resident")
        a, b = compare_schedule(latency), compare_schedule(stationary)
        self.assertLess(b["total_ns"], a["total_ns"])
        self.assertGreater(b["first_frame_ns"], a["first_frame_ns"])
        base = PerformanceScenario()
        fast_clock = compare_schedule(stationary, replace(base, core_hz=2e8))
        fast_link = compare_schedule(stationary, replace(base, tx_bytes_per_second=5e7, rx_bytes_per_second=5e7))
        self.assertLess(fast_link["total_ns"], fast_clock["total_ns"])

    def test_double_buffer_prefetch_and_capacity_plateau(self):
        single = self.compile(target=Target(scratchpad_bytes=128, buffers=1), weight_policy="resident")
        double = self.compile(target=Target(scratchpad_bytes=128, buffers=2), weight_policy="resident")
        larger = self.compile(target=Target(scratchpad_bytes=192, buffers=2), weight_policy="resident")
        a = compare_schedule(single, overlap=True)
        b = compare_schedule(double, overlap=True, capture=True)
        c = compare_schedule(larger, overlap=True)
        self.assertLess(b["total_ns"], a["total_ns"])
        self.assertEqual(b["total_ns"], c["total_ns"])
        self.assertTrue(validate_calendar(b))
        # At least one data transaction is actually simultaneous with compute.
        self.assertTrue(any(min(link["end_ns"], core["end_ns"]) > max(link["start_ns"], core["start_ns"])
                            for link in b["events"] if link["resource"] == "link"
                            for core in b["events"] if core["resource"] == "core"))

    def test_independent_checker_rejects_weight_clobber_during_compute(self):
        manifest = self.compile(rows=4, frames=1, weight_policy="resident")
        scenario = PerformanceScenario(tx_bytes_per_second=1e12, rx_bytes_per_second=1e12,
                    device_setup_cycles=0, host_dispatch_ns=0, vm_offload_call_ns=0,
                    host_pack_ns_per_element=0, host_unpack_ns_per_element=0)
        result = compare_schedule(manifest, scenario, overlap=True, capture=True)
        self.assertTrue(validate_calendar(result))
        weight = next(event for event in result["events"] if event["kind"] == "LOAD_W")
        core = next(event for event in result["events"] if event["resource"] == "core")
        weight["start_ns"] = core["start_ns"]+10
        weight["end_ns"] = weight["start_ns"]+weight["duration_ns"]
        # Remove dependency metadata deliberately: byte-range validation must
        # still catch this physically unsafe cross-resource overlap.
        for event in result["events"]:
            event["dependencies"] = []
        with self.assertRaisesRegex(ValueError, "conflicting scratchpad"):
            validate_calendar(result)

    def test_odd_k_tail_capacity_and_rejections(self):
        for buffers in (1,2):
            manifest = self.compile(rows=7, k=17, frames=3, target=Target(buffers=buffers),
                                    order="weight_stationary", weight_policy="resident")
            self.assertTrue(any(tile["rows"]%2 for tile in manifest["tiles"]))
            self.assertEqual(manifest["predicted_transfers"]["load_w_bytes"], 7*9)
            self.assertTrue(validate_calendar(compare_schedule(manifest, overlap=True, capture=True)))
        with self.assertRaises(ValueError):
            self.compile(target=Target(scratchpad_bytes=8))
        with self.assertRaises(ValueError):
            self.compile(frames=0)
        with self.assertRaises(ValueError):
            self.compile(order="unknown")


LIBRARY = Path(os.environ.get("TINYLPU_RUNTIME_LIBRARY", ROOT/"build_esl/libtinylpu_plan_runtime.so"))


@unittest.skipUnless(LIBRARY.is_file() and importlib.util.find_spec("numpy"), "C++ shared library and numpy required")
class BatchInterpreterTests(unittest.TestCase):
    def test_orders_policies_tail_and_slots_execute_the_same_outputs(self):
        import numpy as np
        from tinylpu_compiler.runtime import PlanSession

        weights = [[((r*3+c)%16)-8 for c in range(17)] for r in range(7)]
        values = np.asarray([[(frame*5+c)%16-8 for c in range(17)] for frame in range(3)], dtype="float32")
        expected = (values @ np.asarray(weights, dtype="float32").T).reshape(-1).tolist()
        for buffers in (1,2):
            for order in ("latency_first", "weight_stationary"):
                for policy in ("reload", "resident"):
                    with tempfile.TemporaryDirectory() as folder:
                        manifest = compile_batch_linear(weights, 3, folder, Target(buffers=buffers),
                                                        order=order, weight_policy=policy)
                        with PlanSession(folder, LIBRARY) as session:
                            report = session.run(values.reshape(1,-1))
                            self.assertEqual(report["result"], expected)
                            self.assertEqual(report["core_cycles"], manifest["predicted_core_cycles"])
                            for name, count in manifest["predicted_transfers"].items():
                                self.assertEqual(report[name], count)


if __name__ == "__main__":
    unittest.main()
