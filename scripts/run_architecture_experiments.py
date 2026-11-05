#!/usr/bin/env python3
"""Actual ISA execution plus separate port-limited microarchitecture analysis."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"compiler"), str(ROOT/"scripts")]
from tinylpu_compiler.isa import IsaRuntime, Profile, bind, oracle, save_plan, encode
from tinylpu_compiler.architecture import ArchRuntime, Resources, choose_plan, architecture_time
from run_isa_experiments import dump, csv_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build_esl/libtinylpu_isa.so")
    parser.add_argument("--architecture-library", default="build_esl/libtinylpu_architecture.so")
    parser.add_argument("--output", default="reports/architecture/resource")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"configs/architecture_experiments.json").read_text())
    rng, runtime, analyzer = np.random.default_rng(config["seed"]), IsaRuntime(args.library), ArchRuntime(args.architecture_library)
    rows, search_rows, factors = [], [], []
    for m,k in config["models"]:
        weights = rng.integers(-8, 8, size=(m,k), dtype=np.int32).tolist()
        all_inputs = rng.integers(-8, 8, size=(8,k), dtype=np.int32).tolist()
        source_hash = hashlib.sha256(json.dumps([weights,all_inputs]).encode()).hexdigest()
        for frames in config["frames"]:
            expected = oracle(weights, all_inputs[:frames], 32, 32)[0]
            for preset in config["presets"]:
                p = Profile(preset["capacity"],preset["lanes"],3,32,1)
                r = Resources(preset["banks"],preset["ports"],1,1)
                plan, report, search = choose_plan(weights,frames,p,r,analyzer,preset["storage"],32)
                actual, numeric = runtime.run(plan,all_inputs[:frames])
                assert actual == expected and numeric["overflow_events"] == 0
                for field in ("command_bytes","macs","load_bytes","store_bytes","mvm_count"):
                    assert report[field] == numeric[field]
                if preset["storage"] == "bank4" and p.lanes == 2:
                    assert report["conflict_cycles"] == 0
                row = {"model":f"linear_{k}x{m}","frames":frames,"preset":preset["name"],
                       "capacity":p.scratch_bytes,"lanes":p.lanes,"banks":r.banks,
                       "storage":preset["storage"],"tile_rows":plan["tile_rows"],"tile_k":plan["tile_k"],
                       "order":plan["order"],"peak_bytes":plan["layout"]["end"],
                       "source_sha256":source_hash,**report,"time_us":architecture_time(report)*1e6}
                row["frames_per_second"] = frames/(row["time_us"]*1e-6)
                last_store=max(s["command"] for s in plan["stores"] if s["frame"]==0)
                prefix=dict(plan,commands_hex=encode(plan["commands"][:last_store+1]+[{"op":"END"}]).hex())
                first=analyzer.analyze(prefix,r)
                first["command_bytes"]-=4
                first["command_count"]-=1
                first["device_cycles"]-=1
                row["frame0_us"]=architecture_time(first)*1e6
                rows.append(row)
                for record in search:
                    search_rows.append({"model":row["model"],"frames":frames,"preset":preset["name"],**record})
                folder=output/"artifacts"/f"{row['model']}_b{frames}_{preset['name']}"
                save_plan(plan,folder)
                (folder/"loads.bin").write_bytes(bind(plan,all_inputs[:frames])[1])
                dump(folder/"execution.json",{"input":all_inputs[:frames],"output":actual,"numeric":numeric,"resource":report})
                for hz in (100e6,150e6,200e6):
                    for link in (1e6,10e6,50e6):
                        for dispatch in (0,1000):
                            factors.append({"model":row["model"],"frames":frames,"preset":preset["name"],
                                            "hz":hz,"link_Bps":link,"dispatch_ns":dispatch,
                                            "time_us":architecture_time(report,hz,link,dispatch)*1e6})
        print(json.dumps({"model":f"linear_{k}x{m}","total_rows":len(rows)}),flush=True)
    csv_file(output/"comparison.csv",rows)
    csv_file(output/"search.csv",search_rows)
    csv_file(output/"factors.csv",factors)
    dump(output/"summary.json",{"executed_configurations":len(rows),"search_candidates":len(search_rows),
                               "factors":len(factors),"all_passed":True,"physical_fmax_validated":False,
                               "numeric_engine":"independent existing ISA C++",
                               "timing_engine":"new C++ byte-bank resource analyzer"})
    print(json.dumps({"rows":len(rows),"all_passed":True}),flush=True)


if __name__ == "__main__": main()
