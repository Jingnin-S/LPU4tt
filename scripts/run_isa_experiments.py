#!/usr/bin/env python3
"""Execute actual candidate command streams and select an integer device ISA."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"compiler"))
from tinylpu_compiler.isa import (IsaRuntime, Profile, bind, compile_linear, encode,
                                 estimate_shape, layout, oracle, predicted, save_plan, time_seconds)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose(weights, frames, profile, bits, output, order, config):
    m, k = len(weights), len(weights[0])
    choices_m = sorted({v for v in (1, 2, 3, 4, 7, 8, 16, m) if v <= min(m, 16)})
    choices_k = sorted({min(k, 64), *[v for v in config["tile_k_candidates"] if v <= min(k, 64)]})
    best, records = None, []
    for tm in choices_m:
        for tk in choices_k:
            try:
                layout(tm, tk, k, bits, output, profile)
            except ValueError:
                continue
            counts = estimate_shape(m, k, frames, tm, tk, bits, output, profile, order)
            cost = time_seconds(counts, config["selection_core_hz"],
                                config["selection_link_Bps"], config["selection_host_ns"])
            records.append({"tile_rows": tm, "tile_k": tk, "time_us": cost*1e6,
                            "peak_bytes": layout(tm, tk, k, bits, output, profile)["end"], **counts})
            key = (cost, counts["load_bytes"], counts["command_bytes"], tm, tk)
            if best is None or key < best[0]:
                best = (key, tm, tk)
    if best is None:
        raise ValueError("No feasible tile")
    _, tm, tk = best
    plan = compile_linear(weights, frames, profile, bits, output, 4, tm, tk, order)
    if estimate_shape(m, k, frames, tm, tk, bits, output, profile, order) != plan["predicted"]:
        raise AssertionError("Search estimate disagrees with actual emitted stream")
    return plan, records


def exact_run(runtime, plan, weights, inputs, expected=None):
    result, report = runtime.run(plan, inputs)
    if expected is None:
        expected = oracle(weights, inputs, plan["profile"]["accumulator_bits"], plan["output_bits"])
    if result != expected[0] or report["overflow_events"] != expected[1] or report["saturation_events"] != expected[2]:
        raise AssertionError("Independent integer oracle mismatch")
    return result, report


def completion_times(plan, config):
    """Full-frame completion after its last STORE, including sequential transport."""
    last = {}
    for spec in plan["stores"]:
        last[spec["frame"]] = spec["command"]
    times = []
    for frame in range(plan["frames"]):
        counts = predicted(plan["commands"][:last[frame]+1]+[{"op": "END"}],
                           Profile(**plan["profile"]), plan["encoding"])
        counts["command_bytes"] -= plan["encoding"]
        for field in ("command_count", "decode_cycles", "device_cycles"):
            counts[field] -= 1
        times.append(time_seconds(counts, config["selection_core_hz"], config["selection_link_Bps"],
                                  config["selection_host_ns"])*1e6)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--legacy-library", required=True)
    parser.add_argument("--config", default=str(ROOT/"configs/isa_experiments.json"))
    parser.add_argument("--output", default=str(ROOT/"reports/isa"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = IsaRuntime(args.library)
    rng = np.random.default_rng(config["seed"])
    all_rows, searches, fixtures, representatives, expected_cache = [], [], [], [], {}
    old_records, diagnostics, mlp_records = [], [], []
    total_runs = 0
    for m, k in config["fixtures"]:
        name = f"linear_{k}x{m}"
        weights = rng.integers(-8, 8, size=(m, k), dtype=np.int32).tolist()
        all_inputs = rng.integers(-8, 8, size=(max(config["frames"]), k), dtype=np.int32).tolist()
        source = json.dumps({"weights": weights, "inputs": all_inputs}, separators=(",", ":")).encode()
        digest = hashlib.sha256(source).hexdigest()
        dump(output/"fixtures"/f"{name}.json", {"weights": weights, "inputs": all_inputs, "source_sha256": digest})
        fixture_runs = 0
        for bits in (4, 8):
            if bits == 8 and [m, k] not in config["int8_fixtures"]:
                continue
            for frames in config["frames"]:
                inputs = all_inputs[:frames]
                for out_bits in config["output_bits"]:
                    expected = oracle(weights, inputs, 32, out_bits)
                    expected_cache[(name, frames, out_bits)] = expected
                    for capacity in config["scratch_bytes"]:
                        profile = Profile(capacity, config["lanes"], config["depth"], 32,
                                          config["memory_byte_bw"], bits == 8)
                        for order in config["orders"]:
                            plan, search = choose(weights, frames, profile, bits, out_bits, order, config)
                            for record in search:
                                searches.append({"fixture": name, "frames": frames, "input_bits": bits,
                                                 "output_bits": out_bits, "scratch_bytes": capacity,
                                                 "order": order, **record})
                            for encoding in config["encodings"]:
                                plan["encoding"] = encoding
                                binary = encode(plan["commands"], encoding)
                                plan["commands_hex"] = binary.hex()
                                plan["commands_sha256"] = hashlib.sha256(binary).hexdigest()
                                plan["predicted"] = predicted(plan["commands"], profile, encoding)
                                result, report = exact_run(runtime, plan, weights, inputs, expected)
                                total_runs += 1
                                fixture_runs += 1
                                row = {"fixture": name, "m": m, "k": k, "frames": frames,
                                       "input_bits": bits, "acc_bits": 32, "output_bits": out_bits,
                                       "scratch_bytes": capacity, "order": order, "encoding": encoding,
                                       "tile_rows": plan["tile_rows"], "tile_k": plan["tile_k"],
                                       "peak_bytes": plan["layout"]["end"], "source_sha256": digest,
                                       **{key: report[key] for key in plan["predicted"]},
                                       "overflow_events": report["overflow_events"],
                                       "saturation_events": report["saturation_events"],
                                       "time_us": time_seconds(report, config["selection_core_hz"],
                                                               config["selection_link_Bps"], config["selection_host_ns"])*1e6}
                                completions = completion_times(plan, config)
                                row["frame0_us"] = completions[0]
                                row["mean_frame_completion_us"] = sum(completions)/len(completions)
                                row["frames_per_second"] = frames/(row["time_us"]*1e-6)
                                all_rows.append(row)
                                if frames == 8 and capacity == 128 and out_bits == 32 and order == "frame_major":
                                    folder = output/"artifacts"/f"{name}_i{bits}_e{encoding}"
                                    save_plan(plan, folder)
                                    commands, loads = bind(plan, inputs)
                                    (folder/"loads.bin").write_bytes(loads)
                                    dump(folder/"execution.json", {"result": result, "report": report, "source_sha256": digest})
                                    representatives.append(row)
        fixtures.append({"name": name, "rows": m, "k": k, "runs": fixture_runs, "source_sha256": digest})
        # Independent old ABI path verifies exactly the same integer source, not new timing.
        if k <= 64:
            from tinylpu_compiler.batch_plan import compile_batch_linear
            from tinylpu_compiler.runtime import PlanSession
            from tinylpu_compiler.target import Target
            for frames in config["frames"]:
                folder = output/"legacy"/f"{name}_b{frames}"
                manifest = compile_batch_linear(weights, frames, folder,
                                               Target(scratchpad_bytes=256), name,
                                               "latency_first", "resident")
                with PlanSession(folder, args.legacy_library) as session:
                    report = session.run(np.asarray(all_inputs[:frames], dtype=np.float32).reshape(1, -1))
                    actual = np.asarray(report["result"])
                expected = expected_cache[(name, frames, 16)][0]
                if actual.reshape(frames, m).tolist() != expected:
                    raise AssertionError("Old ABI cross-check failed")
                old_records.append({"fixture": name, "frames": frames, "source_sha256": digest,
                                    "core_cycles": report["core_cycles"], "passed": True})
        print(json.dumps({"fixture": name, "runs": fixture_runs, "total_runs": total_runs}), flush=True)

    # Full-range extremes and cancellation across block boundaries, all actual streams.
    for bits, k, av, wv, label in ((4, 64, -8, -8, "i4_short"),
                                 (4, 512, -8, -8, "i4_overflow"),
                                 (4, 1024, -8, -8, "i4_long"),
                                 (8, 2, -128, -128, "i8_overflow"),
                                 (8, 4096, -128, -128, "i8_long")):
        for accumulator in (16, 32):
            for out_bits in sorted({16, accumulator}):
                profile = Profile(64, accumulator_bits=accumulator, enable_int8=bits == 8)
                for encoding in (4, 8):
                    weights, inputs = [[wv]*k]*3, [[av]*k]
                    plan = compile_linear(weights, profile=profile, input_bits=bits, output_bits=out_bits,
                                          encoding=encoding, tile_rows=2, tile_k=min(k, 8))
                    result, report = exact_run(runtime, plan, weights, inputs)
                    diagnostics.append({"case": label, "input_bits": bits, "global_k": k,
                                        "acc_bits": accumulator, "output_bits": out_bits,
                                        "encoding": encoding, "mathematical_sum": av*wv*k,
                                        "actual": result[0][0], "overflow_events": report["overflow_events"],
                                        "saturation_events": report["saturation_events"], "passed": True})
                    total_runs += 1
    # Cancellation: partial sums exceed INT16 then return in range; SAT only at final.
    w = [-8]*512+[7]*512
    plan = compile_linear([w], profile=Profile(64), output_bits=16, tile_rows=1, tile_k=16)
    result, report = exact_run(runtime, plan, [w], [[-8]*1024])
    assert result == [[4096]] and report["saturation_events"] == 0
    diagnostics.append({"case": "cross_block_cancellation", "actual": result, "passed": True})
    total_runs += 1

    # Two-layer integer graph: explicit host requant, not a hidden device opcode.
    w1 = rng.integers(-8, 8, size=(8, 16), dtype=np.int32).tolist()
    w2 = rng.integers(-8, 8, size=(4, 8), dtype=np.int32).tolist()
    inputs = rng.integers(-8, 8, size=(8, 16), dtype=np.int32).tolist()
    for encoding in (4, 8):
        p1 = compile_linear(w1, frames=8, encoding=encoding)
        y1, r1 = exact_run(runtime, p1, w1, inputs)
        hidden = np.clip(np.rint(np.maximum(y1, 0)/32), 0, 7).astype(np.int32).tolist()
        p2 = compile_linear(w2, frames=8, encoding=encoding)
        y2, r2 = exact_run(runtime, p2, w2, hidden)
        mlp_records.append({"encoding": encoding, "frames": 8, "host_op": "relu; RNE(x/32); clip[0,7]",
                            "source_sha256": hashlib.sha256(json.dumps([w1, w2, inputs]).encode()).hexdigest(),
                            "layer1_report": r1, "layer2_report": r2, "output": y2, "passed": True})
        total_runs += 2
    factors = []
    for row in representatives:
        for hz in config["factor_core_hz"]:
            for bps in config["factor_link_Bps"]:
                for host in config["factor_host_ns"]:
                    value = time_seconds(row, hz, bps, host)
                    factors.append({**row, "core_hz": hz, "link_Bps": bps, "host_ns_per_command": host,
                                    "scenario_time_us": value*1e6, "scenario_frames_per_second": row["frames"]/value})
    csv_file(output/"comparison.csv", all_rows)
    csv_file(output/"tiling_search.csv", searches)
    csv_file(output/"performance_factors.csv", factors)
    dump(output/"diagnostics.json", diagnostics)
    dump(output/"legacy_checks.json", old_records)
    dump(output/"mlp_checks.json", mlp_records)
    dump(output/"config.json", config)
    summary = {"fixtures": fixtures, "comparison_rows": len(all_rows), "tiling_candidates": len(searches),
               "factor_rows": len(factors), "legacy_checks": len(old_records),
               "diagnostic_runs": len(diagnostics), "mlp_graphs": len(mlp_records),
               "cpp_runs": total_runs, "errors": 0, "physical_performance_validated": False}
    dump(output/"summary.json", summary)
    lines = ["# ISA experiment results", "", "Actual encoded streams; independent integer oracle and exact cycle/byte checks.",
             "Timing is a sequential scenario estimate, not measured hardware.", "", "```json", json.dumps(summary, indent=2), "```", "",
             "Primary input values are identical across INT4/INT8 to isolate storage/transport; full-range INT8 is in diagnostics.",
             "Tile selection minimizes modeled time at 100MHz / 10MB/s / zero host dispatch, within the listed candidate set.",
             "Both encodings use the geometry chosen for mixed32/64; no claim of globally optimal tiling.",
             "Internal scratch copy, group drains and partial-sum traffic are counted; no asynchronous overlap.",
             "All frames arrive at t=0; aggregation wait and host quantization time are excluded."]
    (output/"analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
