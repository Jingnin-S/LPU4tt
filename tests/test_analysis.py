"""Independent accounting and failure-contract checks for the ESL analysis runner."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_esl.py"
SPEC = importlib.util.spec_from_file_location("analyze_esl", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def sample(kernel="dot4", reuse=1, wait=0):
    """Test fixtures use independently written closed forms, not runner helpers."""
    m = {"dot4": 1, "gemv2x4": 2}[kernel]
    return {
        "kernel": kernel, "rows": m, "k": 4, "reuse": reuse, "host_wait": wait,
        "functional_result": [12] if m == 1 else [12, -18],
        "cycle_result": [12] if m == 1 else [12, -18],
        "transactions": reuse, "macs": 4 * m * reuse,
        "weight_load_cycles": 2 * m, "input_load_cycles": 2 * reuse,
        "setup_cycles": reuse, "compute_cycles": 5 * m * reuse,
        "readout_cycles": 2 * m * reuse, "host_wait_cycles": wait * reuse,
        "total_cycles": 2 * m + reuse * (3 + wait + 7 * m),
        "peak_macs_per_cycle": 1, "core_macs_per_cycle": 0.8,
    }


def profile(name="sky", area=17954.72):
    return {
        "id": name, "analysis_template": name, "pdk": "test-" + name,
        "template_repository": "https://example.com/" + name,
        "template_commit": "a" * 40, "source_urls": ["https://example.com/source"],
        "action_ref": "test-action", "librelane_version": "test-version",
        "clock_period_ns": 20, "density_pct": 60, "tile_width_um": 161,
        "tile_height_um": area / 161, "tile_die_area_um2": area,
        "physical_performance_validated": False, "assumptions": ["ESL only"],
    }


class AccountingTests(unittest.TestCase):
    def test_known_cycle_budgets_and_golden_results(self):
        for kernel, reuse, wait, expected in (
            ("dot4", 1, 0, 12), ("gemv2x4", 1, 0, 21),
            ("dot4", 16, 0, 162), ("gemv2x4", 16, 0, 276),
            ("dot4", 16, 16, 418), ("gemv2x4", 16, 16, 532),
        ):
            with self.subTest(kernel=kernel, reuse=reuse, wait=wait):
                data = sample(kernel, reuse, wait)
                analysis.validate_model_output(data, kernel, reuse, wait)
                self.assertEqual(analysis.expected_counts(kernel, reuse, wait)["total_cycles"], expected)
        self.assertEqual(analysis.golden_result("dot4"), [12])
        self.assertEqual(analysis.golden_result("gemv2x4"), [12, -18])

    def test_time_throughput_and_disjoint_fractions(self):
        row = analysis.make_metrics(profile(), sample(), 25)
        self.assertEqual(row["core_run_us"], 0.2)
        self.assertEqual(row["total_us"], 0.48)
        self.assertAlmostEqual(row["effective_macs_per_second"], 25_000_000 / 3)
        self.assertEqual(row["active_mac_utilization"], 0.8)
        self.assertAlmostEqual(row["overall_mac_utilization"], 1 / 3)
        self.assertAlmostEqual(sum(row[field] for field in (
            "io_fraction", "host_wait_fraction", "setup_fraction", "core_fraction")), 1)
        waiting = analysis.make_metrics(profile(), sample(reuse=16, wait=16), 25)
        self.assertAlmostEqual(waiting["host_wait_fraction"], 256 / 418)
        self.assertEqual(waiting["amortized_weight_load_cycles_per_transaction"], 0.125)

    def test_frequency_and_template_metadata_do_not_change_cycles(self):
        rows = [analysis.make_metrics(profile(name, area), sample("gemv2x4", 16, 4), f)
                for name, area in (("sky", 17954.72), ("ihp", 31318.36), ("gf", 55711.98))
                for f in (10, 25, 50)]
        analysis.assert_template_equivalence(rows)
        self.assertEqual({row["total_cycles"] for row in rows}, {340})
        self.assertAlmostEqual(rows[2]["effective_macs_per_second"] / rows[0]["effective_macs_per_second"], 5)
        self.assertTrue(all(row["physical_performance_validated"] is False for row in rows))
        rows[-1]["total_us"] *= 0.5
        with self.assertRaisesRegex(analysis.AnalysisError, "Template-dependent"):
            analysis.assert_template_equivalence(rows)

    def test_rejects_corrupt_results_and_nonnumeric_frequency(self):
        for field, value in (("cycle_result", [11]), ("functional_result", [12.0]),
                             ("total_cycles", 11), ("compute_cycles", True),
                             ("core_macs_per_cycle", 1)):
            data = sample()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(analysis.AnalysisError):
                analysis.validate_model_output(data, "dot4", 1, 0)
        for frequency in (0, -1, True, float("nan"), float("inf"), "25"):
            with self.subTest(frequency=frequency), self.assertRaises(analysis.AnalysisError):
                analysis.make_metrics(profile(), sample(), frequency)


class ContractTests(unittest.TestCase):
    def test_full_report_matrix_and_separate_template_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "configs" / "templates"
            config.mkdir(parents=True)
            for name in analysis.TEMPLATES:
                (config / (name + ".json")).write_text(json.dumps(profile(name)), encoding="utf-8")
            executable = root / "test-model"
            executable.touch()

            def model_response(model, kernel, reuse, wait, trace_path=None):
                if trace_path is not None:
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text("cycle,event\n0,start\n5,done\n", encoding="utf-8")
                return sample(kernel, reuse, wait)

            output = root / "reports"
            with patch.object(analysis, "run_model", side_effect=model_response) as runner:
                rows = analysis.analyze(executable, root, output)
            self.assertEqual(runner.call_count, 54)
            self.assertEqual(len(rows), 162)
            self.assertEqual(len(list(output.glob("*/traces/*.csv"))), 6)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["row_count"], 162)
            self.assertFalse(summary["physical_performance_validated"])
            for name in analysis.TEMPLATES:
                data = json.loads((output / name / "model_results.json").read_text(encoding="utf-8"))
                self.assertEqual(len(data["samples"]), 18)
                self.assertTrue((output / name / "performance.csv").is_file())
                self.assertTrue((output / name / "analysis.md").is_file())
            chart = ET.parse(output / "workload_breakdown.svg")
            self.assertEqual(chart.getroot().tag, "{http://www.w3.org/2000/svg}svg")

    def test_cli_nonzero_invalid_json_and_bad_model_data_fail(self):
        cases = (
            subprocess.CompletedProcess([], 2, "", "invalid argument"),
            subprocess.CompletedProcess([], 0, "not JSON", ""),
            subprocess.CompletedProcess([], 0, json.dumps(dict(sample(), total_cycles=8)), ""),
        )
        for response in cases:
            with self.subTest(response=response), patch.object(analysis.subprocess, "run", return_value=response):
                with self.assertRaises(analysis.AnalysisError):
                    analysis.run_model(Path("model"), "dot4", 1, 0)

    def test_cli_missing_executable_timeout_and_missing_trace_fail(self):
        for error in (OSError("missing executable"), subprocess.TimeoutExpired("model", 30)):
            with patch.object(analysis.subprocess, "run", side_effect=error), self.assertRaises(analysis.AnalysisError):
                analysis.run_model(Path("model"), "dot4", 1, 0)
        response = subprocess.CompletedProcess([], 0, json.dumps(sample()), "")
        with tempfile.TemporaryDirectory() as temp, patch.object(analysis.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(analysis.AnalysisError, "trace"):
                analysis.run_model(Path("model"), "dot4", 1, 0, Path(temp) / "trace.csv")

    def test_profiles_require_complete_schema_and_unvalidated_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "configs" / "templates"
            config.mkdir(parents=True)
            for name in analysis.TEMPLATES:
                (config / (name + ".json")).write_text(json.dumps(profile(name)), encoding="utf-8")
            self.assertEqual(len(analysis.load_profiles(root)), 3)
            keyed_sources = dict(profile(), source_urls={"workflow": "https://example.com/workflow"})
            (config / "sky.json").write_text(json.dumps(keyed_sources), encoding="utf-8")
            self.assertEqual(len(analysis.load_profiles(root)), 3)
            invalid = profile()
            del invalid["pdk"]
            (config / "sky.json").write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(analysis.AnalysisError, "missing"):
                analysis.load_profiles(root)
            invalid = dict(profile(), physical_performance_validated=True)
            (config / "sky.json").write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(analysis.AnalysisError, "physical_performance_validated"):
                analysis.load_profiles(root)


if __name__ == "__main__":
    unittest.main()
