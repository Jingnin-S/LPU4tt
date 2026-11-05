#!/usr/bin/env python3
"""Compare batch linear schedules, reuse policies, and resource assumptions.

The driver deliberately keeps the compiler/runtime boundary narrow: each model
is imported from Relax once, each generated batch artifact is checked by the
existing C++ PlanSession, and schedule estimates are obtained from the M3
comparison API.  Full-matrix artifacts are temporary; representative
artifacts and traces are retained together with hashes and JSONL validation
records.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_json(command, timeout=60):
    result = subprocess.run(command, text=True, encoding="utf-8", capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError("Command failed: {}\n{}".format(" ".join(map(str, command)),
                                                            result.stderr[:2000]))
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Command returned invalid JSON: {}\n{}".format(command[0],
                                                                           result.stdout[:1000])) from error


def make_weights(rows, k, phase):
    """Use the same small exact-INT4 construction as the compiler experiments."""
    return [[(row * 5 + column * 3 + phase + row) % 5 - 2
             for column in range(k)] for row in range(rows)]


def build_linear_model(torch, rows, k, weights):
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.w)

    model = Linear()
    model.eval()
    return model


def build_targets(matrix):
    for lanes in matrix["lanes"]:
        for depth in matrix["depth"]:
            for capacity in matrix["capacity_bytes"]:
                for buffers in matrix["buffers"]:
                    yield {
                        "target_id": "comparison-p{}-d{}-{}b-b{}".format(
                            lanes, depth, capacity, buffers),
                        "lanes": lanes,
                        "depth": depth,
                        "a_ports": matrix.get("a_ports", 1),
                        "b_ports": lanes,
                        "scratchpad_bytes": capacity,
                        "buffers": buffers,
                        "max_k": matrix.get("max_k", 64),
                        "max_rows": matrix.get("max_rows", 16),
                    }


def build_descriptors(config, model_specs):
    matrix = config["architecture_matrix"]
    targets = list(build_targets(matrix))
    for model in model_specs:
        for target in targets:
            for frames in matrix["frames"]:
                for order in matrix["orders"]:
                    for policy in matrix["weight_policies"]:
                        yield {
                            "model": model["name"],
                            "rows": model["rows"],
                            "k": model["k"],
                            "frames": frames,
                            "order": order,
                            "weight_policy": policy,
                            "target": target,
                            "sweep_kind": "baseline",
                        }


def descriptor_id(descriptor):
    return json_digest(descriptor)[:16]


def make_scenario(data):
    from tinylpu_compiler.performance import PerformanceScenario

    return PerformanceScenario(**data)


def schedule_summary(schedule, sequence_total=None, overlap_saved=None):
    counts = schedule.get("counts", {})
    components = schedule.get("components_ns", {})
    resource_busy = schedule.get("resource_busy_ns", {})
    return {
        "total_ns": schedule.get("total_ns"),
        "first_frame_ns": schedule.get("first_frame_ns"),
        "mean_frame_completion_ns": schedule.get("mean_frame_completion_ns"),
        "amortized_frame_ns": schedule.get("amortized_frame_ns"),
        "inferences_per_second": schedule.get("inferences_per_second"),
        "effective_macs_per_second": schedule.get("effective_macs_per_second"),
        "core_cycles": counts.get("core_cycles"),
        "macs": counts.get("macs"),
        "load_a_bytes": counts.get("load_a_bytes"),
        "load_w_bytes": counts.get("load_w_bytes"),
        "store_y_bytes": counts.get("store_y_bytes"),
        "device_commands": counts.get("device_commands"),
        "command_bytes": counts.get("command_bytes"),
        "vm_offload_calls": counts.get("vm_offload_calls"),
        "components_ns": components,
        "resource_busy_ns": resource_busy,
        "sequence_total_ns": sequence_total,
        "overlap_saved_ns": overlap_saved if overlap_saved is not None else schedule.get("overlap_saved_ns"),
    }


def expected_from_torch(torch, model, batch):
    with torch.no_grad():
        output = model(torch.tensor(batch, dtype=torch.float32)).reshape(-1).tolist()
    if any(value != round(value) for value in output):
        raise RuntimeError("Reference produced a non-integral output")
    return [int(value) for value in output]


def oracle_gemm(oracle, weights, first_frame):
    rows, k = len(weights), len(weights[0])
    transposed = [weights[row][column] for column in range(k) for row in range(rows)]
    command = [str(oracle), "--op", "gemm", "--shape", "1,{},{}".format(k, rows),
               "--lanes", "2", "--depth", "3", "--a-ports", "1", "--b-ports", "2",
               "--seed", "1", "--a", ",".join(map(str, first_frame)),
               "--b", ",".join(map(str, transposed))]
    result = run_json(command)
    if result.get("golden") != result.get("cycle_result"):
        raise RuntimeError("Integer GEMM oracle self-check failed")
    return result["golden"]


def make_input_batch(model_index, descriptor, run_index):
    seed = (1009 + model_index * 100003 + descriptor["frames"] * 1009 +
            descriptor["k"] * 17 + run_index)
    rng = random.Random(seed)
    return [[rng.randrange(-8, 8) for _ in range(descriptor["k"])]
            for _ in range(descriptor["frames"])]


def validate_batch_artifact(folder, manifest, model, torch, library, model_index, descriptor):
    """Run two independent batches through one loaded C++ session."""
    import numpy as np
    from tinylpu_compiler.runtime import PlanSession

    records = []
    session = PlanSession(folder, library)
    try:
        for run_index in range(2):
            batch = make_input_batch(model_index, descriptor, run_index)
            expected = expected_from_torch(torch, model, batch)
            carrier = np.asarray(batch, dtype="float32").reshape(1, -1)
            actual = session.run(carrier)
            if actual["result"] != expected:
                raise RuntimeError("Batch result mismatch for {} run {}".format(
                    descriptor["model"], run_index))
            if actual["core_cycles"] != manifest["predicted_core_cycles"]:
                raise RuntimeError("core cycle mismatch for {}".format(descriptor["model"]))
            if actual["macs"] != manifest["predicted_macs"]:
                raise RuntimeError("MAC mismatch for {}".format(descriptor["model"]))
            if actual["kernel_count"] != len(manifest["kernels"]):
                raise RuntimeError("kernel count mismatch for {}".format(descriptor["model"]))
            for field, expected_bytes in manifest["predicted_transfers"].items():
                if actual[field] != expected_bytes:
                    raise RuntimeError("{} mismatch for {}".format(field, descriptor["model"]))
            records.append({
                "config_id": descriptor_id(descriptor),
                "run_index": run_index,
                "input_sha256": json_digest(batch),
                "expected_sha256": json_digest(expected),
                "actual_sha256": json_digest(actual["result"]),
                "actual": {
                    key: actual[key] for key in
                    ("core_cycles", "macs", "load_a_bytes", "load_w_bytes",
                     "store_y_bytes", "kernel_count", "host_ops")
                },
                "passed": True,
            })
    finally:
        session.close()
    return records


def representative_match(descriptor, representative):
    capacities = representative["capacity_bytes"]
    if not isinstance(capacities, list):
        capacities = [capacities]
    frames = representative["frames"]
    if not isinstance(frames, list):
        frames = [frames]
    return (descriptor["target"]["lanes"] == representative["lanes"] and
            descriptor["target"]["depth"] == representative["depth"] and
            descriptor["target"]["scratchpad_bytes"] in capacities and
            descriptor["target"]["buffers"] == representative["buffers"] and
            descriptor["frames"] in frames)


def factor_scenarios(config):
    base = dict(config["baseline_scenario"])
    for value in config["representative"]["factor_sweeps"]["command_bytes"]:
        data = dict(base)
        data["device_command_bytes"] = value
        data["name"] = "command-{}B-link10MBps-100MHz".format(value)
        yield "command_bytes", value, data
    for value in config["representative"]["factor_sweeps"]["link_bytes_per_second"]:
        data = dict(base)
        data["tx_bytes_per_second"] = value
        data["rx_bytes_per_second"] = value
        data["name"] = "link-{}Bps-command8B-100MHz".format(value)
        yield "link_bytes_per_second", value, data
    for value in config["representative"]["factor_sweeps"]["core_hz"]:
        data = dict(base)
        data["core_hz"] = value
        data["name"] = "clock-{}Hz-command8B-link10MBps".format(value)
        yield "core_hz", value, data


def csv_row(row):
    result = dict(row)
    for key in ("target", "counts", "components_ns", "resource_busy_ns"):
        if key in result:
            result[key + "_json"] = json.dumps(result.pop(key), sort_keys=True)
    return result


CSV_FIELDS = [
    "config_id", "sweep_kind", "factor", "factor_value", "status", "reason",
    "model", "rows", "k", "frames", "lanes", "depth", "capacity_bytes",
    "buffers", "order", "weight_policy", "target_id", "program_sha256",
    "weights_sha256", "program_bytes", "weights_bytes", "job_count", "tile_count",
    "core_cycles", "macs", "load_a_bytes", "load_w_bytes", "store_y_bytes",
    "device_commands", "command_bytes", "first_frame_ns", "mean_frame_completion_ns",
    "amortized_frame_ns", "total_ns", "sequence_total_ns", "overlap_total_ns",
    "overlap_first_frame_ns", "overlap_mean_frame_completion_ns",
    "overlap_amortized_frame_ns", "overlap_saved_ns",
    "inferences_per_second", "effective_macs_per_second",
    "overlap_inferences_per_second", "overlap_effective_macs_per_second",
    "target_json", "counts_json", "components_ns_json", "resource_busy_ns_json",
    "artifact_dir", "validation_runs", "scenario_name"
]


def make_row(descriptor, status, reason=""):
    target = descriptor["target"]
    return {
        "config_id": descriptor_id(descriptor),
        "sweep_kind": descriptor.get("sweep_kind", "baseline"),
        "factor": "",
        "factor_value": "",
        "status": status,
        "reason": reason,
        "model": descriptor["model"],
        "rows": descriptor["rows"],
        "k": descriptor["k"],
        "frames": descriptor["frames"],
        "lanes": target["lanes"],
        "depth": target["depth"],
        "capacity_bytes": target["scratchpad_bytes"],
        "buffers": target["buffers"],
        "order": descriptor["order"],
        "weight_policy": descriptor["weight_policy"],
        "target_id": target["target_id"],
        "program_sha256": "",
        "weights_sha256": "",
        "program_bytes": "",
        "weights_bytes": "",
        "job_count": "",
        "tile_count": "",
        "core_cycles": "",
        "macs": "",
        "load_a_bytes": "",
        "load_w_bytes": "",
        "store_y_bytes": "",
        "device_commands": "",
        "command_bytes": "",
        "first_frame_ns": "",
        "mean_frame_completion_ns": "",
        "amortized_frame_ns": "",
        "total_ns": "",
        "sequence_total_ns": "",
        "overlap_total_ns": "",
        "overlap_first_frame_ns": "",
        "overlap_mean_frame_completion_ns": "",
        "overlap_amortized_frame_ns": "",
        "overlap_saved_ns": "",
        "inferences_per_second": "",
        "effective_macs_per_second": "",
        "overlap_inferences_per_second": "",
        "overlap_effective_macs_per_second": "",
        "target": target,
        "counts": {},
        "components_ns": {},
        "resource_busy_ns": {},
        "artifact_dir": "",
        "validation_runs": 0,
        "scenario_name": "",
    }


def fill_schedule_row(row, schedule, overlap_schedule, sequence_total,
                      overlap_saved, scenario_name):
    summary = schedule_summary(schedule, sequence_total, overlap_saved)
    overlap_summary = schedule_summary(overlap_schedule) if overlap_schedule else {}
    row.update({key: summary[key] for key in (
        "first_frame_ns", "mean_frame_completion_ns", "amortized_frame_ns",
        "total_ns", "sequence_total_ns", "overlap_saved_ns", "inferences_per_second",
        "effective_macs_per_second", "core_cycles", "macs", "load_a_bytes",
        "load_w_bytes", "store_y_bytes", "device_commands", "command_bytes",
        "components_ns", "resource_busy_ns")})
    row["overlap_total_ns"] = overlap_summary.get("total_ns", "")
    row["overlap_first_frame_ns"] = overlap_summary.get("first_frame_ns", "")
    row["overlap_mean_frame_completion_ns"] = overlap_summary.get(
        "mean_frame_completion_ns", "")
    row["overlap_amortized_frame_ns"] = overlap_summary.get("amortized_frame_ns", "")
    row["overlap_inferences_per_second"] = overlap_summary.get(
        "inferences_per_second", "")
    row["overlap_effective_macs_per_second"] = overlap_summary.get(
        "effective_macs_per_second", "")
    row["scenario_name"] = scenario_name
    row["counts"] = summary
    return row


def pareto_points(rows):
    points = []
    for row in rows:
        if (row["status"] != "accepted" or row["sweep_kind"] != "baseline" or
                row["frames"] != 1 or row["order"] != "latency_first" or
                row["weight_policy"] != "reload"):
            continue
        points.append({
            "config_id": row["config_id"],
            "model": row["model"],
            "capacity_bytes": row["capacity_bytes"],
            "lanes": row["lanes"],
            "depth": row["depth"],
            "buffers": row["buffers"],
            "latency_ns": row["total_ns"],
            "area_proxy": False,
        })
    result = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate or other["model"] != candidate["model"]:
                continue
            no_worse = (other["capacity_bytes"] <= candidate["capacity_bytes"] and
                        other["lanes"] <= candidate["lanes"] and
                        other["latency_ns"] <= candidate["latency_ns"])
            strictly_better = (other["capacity_bytes"] < candidate["capacity_bytes"] or
                               other["lanes"] < candidate["lanes"] or
                               other["latency_ns"] < candidate["latency_ns"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/comparison")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/comparison_experiments.json")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Optional development limit; zero runs the configured matrix")
    parser.add_argument("--models", default="",
                        help="Optional comma-separated model names")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from tinylpu_compiler import Target
    from tinylpu_compiler.comparison import compare_schedule, validate_calendar
    from tinylpu_compiler.frontend import import_model
    from tinylpu_compiler.performance import estimate_sequence
    from tinylpu_compiler.plan import verify_artifact
    from tinylpu_compiler.batch_plan import compile_batch_linear

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_snapshot.json", config)
    selected_names = {name for name in args.models.split(",") if name}
    model_specs = [spec for spec in config["models"]
                   if not selected_names or spec["name"] in selected_names]
    if not model_specs:
        raise ValueError("No selected models")
    if args.max_configs < 0:
        raise ValueError("--max-configs must be nonnegative")

    # Import each real Relax model once. The resulting Tiny IR node supplies
    # the static weight matrix for every batch schedule variant.
    model_data = {}
    model_dir = output / "models"
    for model_index, spec in enumerate(model_specs):
        weights = make_weights(spec["rows"], spec["k"], spec["phase"])
        model = build_linear_model(torch, spec["rows"], spec["k"], weights)
        frontend_dir = model_dir / spec["name"] / "frontend"
        node, metadata = import_model(
            model, (torch.zeros(1, spec["k"], dtype=torch.float32),), frontend_dir)
        if node.op != "gemv" or node.source.op != "input":
            raise RuntimeError("Expected one imported linear GEMV for {}".format(spec["name"]))
        imported_weights = [[int(value) for value in row] for row in node.weights]
        if imported_weights != weights:
            raise RuntimeError("Relax-imported weights differ for {}".format(spec["name"]))
        first_frame = make_input_batch(model_index, {"frames": 1, "k": spec["k"]}, 0)[0]
        oracle_result = oracle_gemm(args.oracle.resolve(), weights, first_frame)
        expected_oracle = expected_from_torch(torch, model, [first_frame])
        if oracle_result != expected_oracle:
            raise RuntimeError("Oracle mismatch for {}".format(spec["name"]))
        write_json(model_dir / spec["name"] / "comparison_metadata.json", {
            **metadata,
            "model": spec,
            "weights_sha256": json_digest(weights),
            "oracle_first_frame": {"input": first_frame, "result": oracle_result},
        })
        model_data[spec["name"]] = {
            "spec": spec,
            "model": model,
            "weights": imported_weights,
            "metadata": metadata,
            "oracle_first_frame": {"input": first_frame, "result": oracle_result},
        }

    baseline_scenario_data = config["baseline_scenario"]
    baseline_scenario = make_scenario(baseline_scenario_data)
    rows = []
    validation_records = []
    summary = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0, "factor_rows": 0}
    trace_counts = {name: 0 for name in model_data}
    representative = config["representative"]
    trace_policy = config["trace_policy"]
    validation_path = output / "validation_records.jsonl"
    with validation_path.open("w", encoding="utf-8") as validation_stream:
        for descriptor_index, descriptor in enumerate(build_descriptors(config, model_specs)):
            if args.max_configs and descriptor_index >= args.max_configs:
                break
            summary["total"] += 1
            model_info = model_data[descriptor["model"]]
            model_spec = model_info["spec"]
            row = make_row(descriptor, "rejected")
            config_id = descriptor_id(descriptor)
            try:
                target = Target(**descriptor["target"])
                with tempfile.TemporaryDirectory(prefix="tinylpu-comparison-") as temporary:
                    artifact_dir = Path(temporary)
                    try:
                        manifest = compile_batch_linear(
                            model_info["weights"], descriptor["frames"], artifact_dir, target,
                            descriptor["model"] + "_" + config_id, descriptor["order"],
                            descriptor["weight_policy"])
                    except ValueError as error:
                        row["reason"] = str(error)
                        summary["rejected"] += 1
                        rows.append(row)
                        continue
                    manifest_path = artifact_dir / "manifest.json"
                    if not manifest_path.exists():
                        write_json(manifest_path, manifest)
                    verify_artifact(artifact_dir)
                    required = ("frames", "output_rows", "k", "order", "weight_policy",
                                "jobs", "tiles", "kernels", "predicted_core_cycles",
                                "predicted_macs", "predicted_transfers")
                    missing = [field for field in required if field not in manifest]
                    if missing:
                        raise RuntimeError("Batch manifest missing fields: " + ",".join(missing))
                    row.update({
                        "status": "accepted",
                        "reason": "",
                        "program_sha256": sha256_file(artifact_dir / "program.tlp"),
                        "weights_sha256": sha256_file(artifact_dir / "weights.bin"),
                        "program_bytes": (artifact_dir / "program.tlp").stat().st_size,
                        "weights_bytes": (artifact_dir / "weights.bin").stat().st_size,
                        "job_count": len(manifest["jobs"]),
                        "tile_count": len(manifest["tiles"]),
                    })
                    sequence = compare_schedule(manifest, baseline_scenario,
                                                overlap=False, capture=False)
                    overlap = compare_schedule(manifest, baseline_scenario,
                                               overlap=True, capture=False)
                    sequence_reference = estimate_sequence(
                        [{"kind": "device_plan", "manifest": manifest}], baseline_scenario)
                    delta = abs(sequence["total_ns"] - sequence_reference["total_ns"])
                    if delta > max(1e-6, abs(sequence_reference["total_ns"]) * 1e-9):
                        raise RuntimeError("Sequential schedule differs from estimate_sequence by {} ns".format(delta))
                    overlap_saved = sequence["total_ns"] - overlap["total_ns"]
                    fill_schedule_row(row, sequence, overlap,
                                      sequence_reference["total_ns"], overlap_saved,
                                      baseline_scenario.name)
                    validation = validate_batch_artifact(
                        artifact_dir, manifest, model_info["model"], torch, args.library.resolve(),
                        list(model_data).index(descriptor["model"]), descriptor)
                    row["validation_runs"] = len(validation)
                    validation_records.extend(validation)
                    for record in validation:
                        validation_stream.write(json.dumps({
                            "descriptor": descriptor,
                            "manifest": {
                                "program_sha256": row["program_sha256"],
                                "weights_sha256": row["weights_sha256"],
                                "predicted_core_cycles": manifest["predicted_core_cycles"],
                                "predicted_macs": manifest["predicted_macs"],
                                "predicted_transfers": manifest["predicted_transfers"],
                            },
                            "record": record,
                        }, sort_keys=True) + "\n")
                    summary["accepted"] += 1
                    row["artifact_dir"] = ""
                    if representative_match(descriptor, representative):
                        artifact_keep = output / "artifacts" / descriptor["model"] / config_id
                        shutil.copytree(artifact_dir, artifact_keep, dirs_exist_ok=True)
                        row["artifact_dir"] = str(artifact_keep.relative_to(output))
                        factor_descriptor = descriptor
                        if (descriptor["order"] == representative["factor_order"] and
                                descriptor["weight_policy"] == representative["factor_policy"]):
                            for factor, value, scenario_data in factor_scenarios(config):
                                scenario = make_scenario(scenario_data)
                                factor_seq = compare_schedule(manifest, scenario,
                                                               overlap=False, capture=False)
                                factor_overlap = compare_schedule(manifest, scenario,
                                                                  overlap=True, capture=False)
                                factor_row = make_row(factor_descriptor, "accepted")
                                factor_row["sweep_kind"] = "factor"
                                factor_row["factor"] = factor
                                factor_row["factor_value"] = value
                                factor_row.update({
                                    "program_sha256": row["program_sha256"],
                                    "weights_sha256": row["weights_sha256"],
                                    "program_bytes": row["program_bytes"],
                                    "weights_bytes": row["weights_bytes"],
                                    "job_count": row["job_count"],
                                    "tile_count": row["tile_count"],
                                    "validation_runs": row["validation_runs"],
                                    "artifact_dir": row["artifact_dir"],
                                })
                                fill_schedule_row(
                                    factor_row, factor_seq, factor_overlap, "",
                                    factor_seq["total_ns"] - factor_overlap["total_ns"],
                                    scenario.name)
                                rows.append(factor_row)
                                summary["factor_rows"] += 1
                    if (representative_match(descriptor, trace_policy) and
                            descriptor["order"] == trace_policy["order"] and
                            descriptor["weight_policy"] == trace_policy["weight_policy"] and
                            trace_counts[descriptor["model"]] < trace_policy["per_model"]):
                        trace_dir = output / "traces"
                        sequential_trace = compare_schedule(manifest, baseline_scenario,
                                                             overlap=False, capture=True)
                        overlap_trace = compare_schedule(manifest, baseline_scenario,
                                                        overlap=True, capture=True)
                        validate_calendar(sequential_trace)
                        validate_calendar(overlap_trace)
                        write_json(trace_dir / (descriptor["model"] + "_" + config_id + "_sequential.json"),
                                   sequential_trace)
                        write_json(trace_dir / (descriptor["model"] + "_" + config_id + "_overlap.json"),
                                   overlap_trace)
                        trace_counts[descriptor["model"]] += 1
            except Exception:
                summary["errors"] += 1
                raise
            rows.append(row)
            if (descriptor_index + 1) % 100 == 0:
                print("processed {} configs: accepted={}, rejected={}".format(
                    descriptor_index + 1, summary["accepted"], summary["rejected"]))

    pareto = pareto_points(rows)
    write_csv(output / "comparison.csv", [csv_row(row) for row in rows], CSV_FIELDS)
    factor_rows = [csv_row(row) for row in rows if row["sweep_kind"] == "factor"]
    factor_fields = ["model", "frames", "capacity_bytes", "lanes", "depth", "buffers",
                     "order", "weight_policy", "factor", "factor_value", "scenario_name",
                     "total_ns", "first_frame_ns", "mean_frame_completion_ns",
                     "amortized_frame_ns", "inferences_per_second", "effective_macs_per_second",
                     "overlap_total_ns", "overlap_first_frame_ns",
                     "overlap_mean_frame_completion_ns", "overlap_amortized_frame_ns",
                     "overlap_inferences_per_second", "overlap_effective_macs_per_second",
                     "overlap_saved_ns"]
    write_csv(output / "matched_comparisons.csv", factor_rows, factor_fields)
    write_csv(output / "pareto.csv", pareto,
              ["model", "config_id", "capacity_bytes", "lanes", "depth", "buffers",
               "latency_ns", "area_proxy"])

    model_summaries = []
    for name, info in model_data.items():
        model_summaries.append({
            "name": name,
            "rows": info["spec"]["rows"],
            "k": info["spec"]["k"],
            "weights_sha256": json_digest(info["weights"]),
            "oracle_first_frame": info["oracle_first_frame"],
            "frontend": info["metadata"],
        })
    result = {
        "schema": 1,
        "models": model_summaries,
        "summary": summary,
        "rows_file": "comparison.csv",
        "factor_rows_file": "matched_comparisons.csv",
        "pareto_file": "pareto.csv",
        "validation_records_file": "validation_records.jsonl",
        "physical_performance_validated": False,
        "mlp_cross_layer_weight_stationary": "excluded; this comparison covers linear subgraphs only",
        "policy_notes": {
            "resident": "only legal when the single weight bank can retain the current tile",
            "reload": "reference cold/reload policy; every required tile load remains visible",
            "overlap": "calendar-model assumption; no VM/C ABI DMA is claimed",
            "pareto": "capacity/lanes/latency candidates only; capacity is not physical area",
        },
    }
    write_json(output / "results.json", result)

    accepted = [row for row in rows if row["status"] == "accepted" and row["sweep_kind"] == "baseline"]
    lines = [
        "# M3 batch linear schedule comparison", "",
        "真实Relax导入模型：{}。MLP跨层weight-stationary排除。".format(
            ", ".join(name for name in model_data)), "",
        "| 指标 | 数量 |", "| --- | ---: |",
        "| 架构/批次配置总数 | {} |".format(summary["total"]),
        "| 可行配置 | {} |".format(summary["accepted"]),
        "| 拒绝配置 | {} |".format(summary["rejected"]),
        "| 错误 | {} |".format(summary["errors"]),
        "| 单因素比较行 | {} |".format(summary["factor_rows"]),
        "| 可行配置验证 runs | {} |".format(len(validation_records)), "",
        "基线为100MHz、双向10MB/s、8B分析命令；顺序调度与旧estimate_sequence逐配置核对。",
        "overlap结果是带host/link/core资源约束的日历模型，不代表VM或C ABI已经实现DMA重叠。",
        "每个可行artifact使用同一C++ PlanSession运行两个不同batch输入，并与独立PyTorch matmul对照；每个模型另做一次整数GEMM oracle校验。",
        "Pareto仅表示capacity/lanes/latency候选，不称为面积或物理FMax。", "",
        "边界：所有batch frame在t=0已到达，聚合等待不计；weight-stationary仍报告首帧完成时间。",
        "2B/8B/32B命令与链路值是敏感性假设，不代表真实可编码的host descriptor或冻结ISA；尤其2B不能据此推荐硬件实现。",
        "单W bank、双A/Y buffer和共享字节访问是资源模型假设，不等于真实存储端口；D2/3/4只改变周期模型，不证明200MHz可达。", "",
        "tile_rows采用当前实现的max-legal greedy策略，未穷举所有tile划分；容量变化可能同时改变tile与A地址复用，因此曲线不是全局最优，也不能把非单调性直接归因于硬件存储。", "",
        "## 代表性基线（首帧=1、reload、latency_first）", "",
        "| 模型 | capacity | lanes | depth | buffers | total ns | first frame ns | amortized frame ns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in accepted:
        if (row["frames"] == 1 and row["order"] == "latency_first" and
                row["weight_policy"] == "reload" and
                row["lanes"] == representative["lanes"] and
                row["depth"] == representative["depth"] and
                row["buffers"] == representative["buffers"] and
                row["capacity_bytes"] in representative["capacity_bytes"]):
            lines.append("| {model} | {capacity_bytes} | {lanes} | {depth} | {buffers} | {total_ns:.3f} | {first_frame_ns:.3f} | {amortized_frame_ns:.3f} |".format(**row))
    lines.extend(["", "单因素匹配结果见 `matched_comparisons.csv`；代表trace见 `traces/`；每配置二次输入验证见 `validation_records.jsonl`。", ""])
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Comparison experiment PASS: {} accepted, {} rejected, {} validation runs; {}".format(
        summary["accepted"], summary["rejected"], len(validation_records), output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print("Comparison experiment failed: {}".format(error), file=sys.stderr)
        sys.exit(1)

