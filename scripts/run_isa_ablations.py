#!/usr/bin/env python3
"""Controlled instruction-necessity experiments, using executed byte streams."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"compiler"), str(ROOT/"scripts")]
from run_isa_experiments import csv_file, dump, exact_run
from tinylpu_compiler.isa import (IsaRuntime, Profile, bind, compile_linear, encode,
                                 predicted, save_plan, time_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", default=str(ROOT/"reports/isa/ablations"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime, rows = IsaRuntime(args.library), []

    def record(name, plan, report, detail=None):
        plan["name"] = name
        plan["commands_sha256"] = hashlib.sha256(bytes.fromhex(plan["commands_hex"])).hexdigest()
        rows.append({"case": name, **{k: report[k] for k in plan["predicted"]},
                     "time_us": time_seconds(report)*1e6})
        save_plan(plan, output/name)
        (output/name/"loads.bin").write_bytes(bind(plan, inputs)[1])
        dump(output/name/"execution.json", {"report": report, "detail": detail})

    w = [[(r*5+i*3) % 16-8 for i in range(16)] for r in range(8)]
    inputs = [[(f*3+i*7) % 16-8 for i in range(16)] for f in range(8)]
    base = compile_linear(w, 8, output_bits=32, tile_rows=8, tile_k=16)
    for name, plan in (("fused_rows_resident", base),
                       ("reload_weights", compile_linear(w, 8, tile_rows=8, tile_k=16, reuse_weights=False))):
        result, report = exact_run(runtime, plan, w, inputs)
        record(name, plan, report, result)
    unrolled = copy.deepcopy(base)
    commands, mapping = [], {}
    for index, command in enumerate(base["commands"]):
        mapping[index] = len(commands)
        if command["op"] == "MVM":
            for start in range(0, command["rows"], 2):
                commands.append({**command, "rows": min(2, command["rows"]-start),
                                 "w": command["w"]+start*command["stride"], "y": command["y"]+start*4})
        else:
            commands.append(command)
    unrolled["commands"] = commands
    unrolled["commands_hex"] = encode(commands).hex()
    unrolled["predicted"] = predicted(commands, Profile(), 4)
    for spec in unrolled["load_specs"]+unrolled["stores"]:
        spec["command"] = mapping[spec["command"]]
    result, report = exact_run(runtime, unrolled, w, inputs)
    record("unrolled_row_groups_same_data", unrolled, report, result)

    # Host reduction alternative: each K chunk returns raw partials over the link.
    w = [[(r*5+i*3) % 16-8 for i in range(256)] for r in range(8)]
    inputs = [[(f*3+i*7) % 16-8 for i in range(256)] for f in range(8)]
    base = compile_linear(w, 8, output_bits=32, tile_rows=2, tile_k=64)
    result, report = exact_run(runtime, base, w, inputs)
    record("device_accumulate", base, report, result)
    commands, partials, job, mapping = [], [], 0, {}
    jobs = base["stores"]
    for index, command in enumerate(base["commands"]):
        mapping[index] = len(commands)
        if command["op"] == "MVM":
            commands.append({**command, "accumulate": False})
            commands.append({"op": "STORE", "addr": command["y"], "count": command["rows"]*4})
            partials.append({**jobs[job], "command": len(commands)-1})
        elif command["op"] == "STORE":
            job += 1
        else:
            commands.append(command)
    alt = copy.deepcopy(base)
    alt["commands"] = commands
    alt["commands_hex"] = encode(commands).hex()
    alt["predicted"] = predicted(commands, Profile(), 4)
    alt["stores"] = partials
    for spec in alt["load_specs"]:
        spec["command"] = mapping[spec["command"]]
    encoded, loads = bind(alt, inputs)
    report = runtime.run_raw(encoded, loads)
    for field, value in alt["predicted"].items():
        assert report[field] == value
    payload = bytes.fromhex(report["stores_hex"])
    actual, cursor = [[0]*8 for _ in range(8)], 0
    for spec in partials:
        for row in range(spec["rows"]):
            actual[spec["frame"]][spec["row"]+row] += int.from_bytes(payload[cursor:cursor+4], "little", signed=True)
            cursor += 4
    assert cursor == len(payload) and actual == result
    record("host_partial_sum_same_chunks", alt, report,
           {"output": actual, "host_work": "sum raw K partials; CPU reduction time excluded",
            "note": "artifact requires host sum rather than normal unpack_stores"})
    csv_file(output/"comparison.csv", rows)
    dump(output/"summary.json", {"executed_streams": len(rows), "rows": rows, "all_passed": True,
                                "scenario": "100MHz,10MB/s,zero host dispatch,zero host sum cost"})
    print(json.dumps({"ablation_streams": len(rows), "all_passed": True}), flush=True)


if __name__ == "__main__":
    main()
