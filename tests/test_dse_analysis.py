import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("explore_design_space", ROOT / "scripts/explore_design_space.py")
dse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dse)


def architecture(lanes=1, depth=2, buffers=1, drain=True):
    return {"id": "test", "lanes": lanes, "depth": depth, "buffers": buffers, "row_drain": drain}


def workload(rows=1, k=4, frames=1):
    return {"id": "test", "rows": rows, "k": k, "frames": frames}


class DseTests(unittest.TestCase):
    def test_v1_end_to_end_equivalence(self):
        for rows, expected in ((1, 12), (2, 21)):
            r = dse.schedule(workload(rows), architecture(), 32, 25, 25)
            self.assertEqual(r["total_cycles"], expected)
            self.assertEqual(r["compute_cycles"], rows * 5)
            self.assertEqual(r["overlap_cycles"], 0)

    def test_core_continuous_pipeline_and_masked_lanes(self):
        self.assertEqual(dse.core_cycles(5, 8, 2, 3), 26)
        self.assertEqual(dse.core_cycles(5, 8, 2, 3, True), 30)
        self.assertEqual(dse.core_cycles(1, 4, 4, 4), 7)

    def test_capacity_includes_padding_and_double_buffers(self):
        self.assertEqual(dse.storage_bytes(1, 4, 4, 1), 12)
        self.assertEqual(dse.storage_bytes(1, 4, 4, 2), 16)
        w = workload(8, 32, 16)
        a = architecture(lanes=4, depth=3, buffers=2)
        self.assertEqual(dse.make_tiles(w, a, 64), [])
        self.assertTrue(dse.make_tiles(w, a, 128))
        for size in dse.make_tiles(w, a, 128):
            self.assertLessEqual(dse.storage_bytes(size, 32, 4, 2), 128)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            dse.schedule(w, a, 64, 100, 25)

    def test_independent_link_bandwidth_limits_clock_scaling(self):
        a, w = architecture(depth=3, drain=False), workload()
        low = dse.schedule(w, a, 32, 100, 25)
        high = dse.schedule(w, a, 32, 200, 25)
        self.assertEqual(low["transferred_bytes"], 7)
        self.assertAlmostEqual(low["io_cycles"] / 100, high["io_cycles"] / 200)
        self.assertGreater(high["effective_mmacs_per_second"], low["effective_mmacs_per_second"])
        self.assertLess(high["effective_mmacs_per_second"] / low["effective_mmacs_per_second"], 1.2)

    def test_ping_pong_has_legal_overlap_and_resource_ownership(self):
        w = workload(4, 16, 16)
        a = architecture(lanes=2, depth=3, buffers=2, drain=False)
        r = dse.schedule(w, a, 128, 100, 25, capture=True)
        serial = dse.schedule(w, dict(a, buffers=1), 128, 100, 25)
        self.assertLess(r["total_cycles"], serial["total_cycles"])
        self.assertGreater(r["overlap_cycles"], 0)
        events = r["trace"]
        for resource in ("io", "compute"):
            ordered = sorted((e for e in events if e["resource"] == resource), key=lambda e: e["start_tick"])
            for old, new in zip(ordered, ordered[1:]):
                self.assertLessEqual(old["end_tick"], new["start_tick"])
        for job in range(r["jobs"]):
            chunks = {e["kind"]: e for e in events if e["job"] == job}
            self.assertLessEqual(chunks["input"]["end_tick"], chunks["compute"]["start_tick"])
            self.assertLessEqual(chunks["compute"]["end_tick"], chunks["output"]["start_tick"])
        for slot in range(2):
            jobs = sorted({e["job"] for e in events if e["slot"] == slot})
            for old, new in zip(jobs, jobs[1:]):
                old_end = next(e["end_tick"] for e in events if e["job"] == old and e["kind"] == "output")
                new_start = next(e["start_tick"] for e in events if e["job"] == new and e["kind"] == "input")
                self.assertLessEqual(old_end, new_start)

    def test_weight_switch_does_not_clobber_readers(self):
        w, a = workload(8, 32, 4), architecture(lanes=2, depth=4, buffers=2, drain=False)
        r = dse.schedule(w, a, 128, 100, 25, capture=True)
        self.assertGreater(r["tiles"], 1)
        for weight in (e for e in r["trace"] if e["kind"] == "weight"):
            for compute in (e for e in r["trace"] if e["kind"] == "compute"):
                if compute["tile"] != weight["tile"]:
                    self.assertTrue(compute["end_tick"] <= weight["start_tick"] or
                                    compute["start_tick"] >= weight["end_tick"])

    def test_weight_stationary_trades_latency_for_transfer_volume(self):
        w, a = workload(8, 32, 16), architecture(depth=3, drain=False)
        weight = dse.schedule(w, a, 64, 100, 25, "weight_stationary")
        latency = dse.schedule(w, a, 64, 100, 25, "latency_first")
        self.assertEqual(weight["weight_transfer_bytes"], 128)
        self.assertEqual(latency["weight_transfer_bytes"], 128 * 16)
        self.assertLess(weight["total_cycles"], latency["total_cycles"])
        self.assertGreater(weight["first_frame_us"], latency["first_frame_us"])

    def test_no_free_fir_window_reuse_or_lane_speedup(self):
        w = workload(1, 8, 16)
        scalar = dse.schedule(w, architecture(depth=3, drain=False), 64, 100, 25)
        wide = dse.schedule(w, architecture(lanes=4, depth=3, drain=False), 64, 100, 25)
        self.assertEqual(scalar["total_cycles"], wide["total_cycles"])
        self.assertEqual(scalar["transferred_bytes"], 4 + 16 * (4 + 1 + 2))
        self.assertAlmostEqual(wide["compute_mac_utilization"], scalar["compute_mac_utilization"] / 4)


if __name__ == "__main__":
    unittest.main()
