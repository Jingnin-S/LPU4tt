#!/usr/bin/env python3
"""Exercise actual TVM VM partitions and keep software timing separate from estimates."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))
from tinylpu_compiler import Target
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.performance import PerformanceScenario, estimate_sequence
from tinylpu_compiler.plan import verify_artifact
from tinylpu_compiler.vm import VmProgram
from run_compiler_experiments import build_models, model_oracle, write_json


def scenarios(config):
    return [PerformanceScenario(name="hypothetical-{}MHz-{}MBps".format(int(hz/1e6), int(rate/1e6)),
                core_hz=hz, tx_bytes_per_second=rate, rx_bytes_per_second=rate, **config["base"])
            for hz in config["core_hz"] for rate in config["bidirectional_bytes_per_second"]]


def timing_summary(samples):
    return {name: {"median_ns": statistics.median(row[name] for row in samples),
                   "min_ns": min(row[name] for row in samples), "max_ns": max(row[name] for row in samples)}
            for name in samples[0]}


def validate_reports(program):
    manifests = [segment["manifest"] for segment in program.segments if segment["kind"] == "device_plan"]
    if len(manifests) != len(program.last_reports):
        raise RuntimeError("Device partition count mismatch")
    for manifest, report in zip(manifests, program.last_reports):
        for field, predicted in {"core_cycles": manifest["predicted_core_cycles"],
                                 "macs": manifest["predicted_macs"],
                                 "host_ops": manifest["predicted_host_ops"],
                                 "kernel_count": len(manifest["kernels"]),
                                 **manifest["predicted_transfers"]}.items():
            if report[field] != predicted:
                raise RuntimeError("Device {} accounting mismatch".format(field))
        if report["host_ops"] != 0:
            raise RuntimeError("VM-native quantization was incorrectly sent to device plan")


def invalid_inputs(program, k, np):
    cases = {"wrong_shape": np.zeros((k,), dtype="float32"),
             "wrong_length": np.zeros((1,k+1), dtype="float32"),
             "wrong_dtype": np.zeros((1,k), dtype="float64"),
             "out_of_range": np.full((1,k), 8, dtype="float32"),
             "fractional": np.full((1,k), 0.5, dtype="float32"),
             "nan": np.full((1,k), float("nan"), dtype="float32"),
             "inf": np.full((1,k), float("inf"), dtype="float32")}
    records = []
    count = program.total_callback_calls
    for name, value in cases.items():
        try:
            program.run(value)
        except ValueError as error:
            records.append({"case": name, "error": str(error)})
        else:
            raise RuntimeError("Invalid VM input accepted: " + name)
    if count != program.total_callback_calls:
        raise RuntimeError("Rejected input unexpectedly reached the device")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/runtime")
    parser.add_argument("--target", type=Path, default=ROOT/"configs/compiler_target.json")
    parser.add_argument("--performance", type=Path, default=ROOT/"configs/runtime_performance.json")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations < 3:
        parser.error("At least 3 measured iterations are required")
    import numpy as np
    import torch
    import tvm

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = Target(**json.loads(args.target.read_text(encoding="utf-8")))
    config = json.loads(args.performance.read_text(encoding="utf-8"))
    write_json(output/"performance_config.json", config)
    rows = []
    rng = random.Random(107)
    for name, model, k in build_models(torch):
        folder = output/name
        start = time.perf_counter_ns()
        node, metadata = import_model(model, (torch.zeros(1,k,dtype=torch.float32),), folder/"frontend")
        import_ns = time.perf_counter_ns()-start
        values_list = [[0]*k, [-8]*k, [7]*k, [-8 if i%2 else 7 for i in range(k)]]
        values_list += [rng.choices(range(-8,8), k=k) for _ in range(28)]
        with VmProgram(node, folder/"vm", args.library, target) as program:
            samples = []
            for values in values_list:
                value = np.asarray([values], dtype="float32")
                before = program.total_callback_calls
                actual = program.run(value).reshape(-1).tolist()
                with torch.no_grad():
                    expected = model(torch.from_numpy(value)).reshape(-1).tolist()
                existing = model_oracle(args.oracle.resolve(), model, values)
                if actual != expected or actual != existing:
                    raise RuntimeError("VM numerical mismatch: " + name)
                if program.total_callback_calls-before != len(program.sessions):
                    raise RuntimeError("VM offload call count mismatch")
                validate_reports(program)
                samples.append({"input": values, "result": actual, "torch_result": expected,
                                "existing_model_result": existing, "offload_reports": program.last_reports,
                                "software_timing": program.last_timing})
            if len({tuple(sample["result"]) for sample in samples}) < 2:
                raise RuntimeError("Input-variation check failed")
            negatives = invalid_inputs(program, k, np)
            # Exercise a callback exception through the real VM, bypassing only
            # the outer Python guard. The callback must enforce INT4 as well.
            try:
                program.vm["main"](tvm.runtime.tensor(np.full((1,k), 8, dtype="float32")))
            except Exception as error:
                negatives.append({"case": "vm_callback_out_of_range", "error": str(error)[:800]})
            else:
                raise RuntimeError("VM callback accepted out-of-range input")
            if program.run(np.asarray([values_list[-1]], dtype="float32")).reshape(-1).tolist() != samples[-1]["result"]:
                raise RuntimeError("VM error recovery failed")
            for _ in range(3):
                program.run(np.asarray([values_list[-1]], dtype="float32"))
            timings = []
            for _ in range(args.iterations):
                program.run(np.asarray([values_list[-1]], dtype="float32"))
                timings.append(program.last_timing.copy())
            estimates = [estimate_sequence(program.segments, scenario) for scenario in scenarios(config)]
            baseline = next(item for item in estimates if item["scenario"]["core_hz"] == 100_000_000
                            and item["scenario"]["tx_bytes_per_second"] == 10_000_000)
            for session in program.sessions:
                verify_artifact(session.folder)
            write_json(folder/"samples.json", {"samples": samples, "negative_cases": negatives})
            write_json(folder/"software_timing.json", {"warmup_runs": 3, "measured_runs": args.iterations,
                       "samples": timings, "summary": timing_summary(timings), "import_wall_ns": import_ns,
                       "vm_build_including_artifact_load_wall_ns": program.build_wall_ns,
                       "artifact_load_wall_ns": program.artifact_load_wall_ns,
                       "interpretation": "CPU simulator/FFI/VM wall time; not silicon latency"})
            write_json(folder/"performance_estimates.json", estimates)
            write_json(folder/"execution_segments.json", program.segments)
            row = {"experiment": name, "runtime_inputs": len(samples), "negative_cases": len(negatives),
                   "byoc_functions": metadata["byoc_functions"], "vm_offload_partitions": len(program.sessions),
                   "host_quant_elements": program.host_quant_elements, "baseline_estimate": baseline,
                   "software_timing": timing_summary(timings), "all_checks_passed": True}
            write_json(folder/"result.json", row)
            rows.append(row)
    write_json(output/"results.json", {"experiments": rows, "torch_version": torch.__version__,
               "tvm_version": tvm.__version__, "target": target.to_dict(),
               "library": str(args.library.resolve()), "oracle": str(args.oracle.resolve()),
               "runtime_inputs": sum(row["runtime_inputs"] for row in rows),
               "negative_cases": sum(row["negative_cases"] for row in rows),
               "execution_boundary": "explicit Tiny IR -> executable Relax VM glue -> packed callback -> C++ shared library; native Relax CPU quantization",
               "generic_RunCodegen_plugin": False, "physical_performance_validated": False})
    lines = ["# TVM VM/runtime 与端到端性能实验", "",
             "真实路径：torch.export/Relax BYOC分区 → Tiny IR → 可执行Relax VM glue → 每GEMV packed callback → 进程内C++/DseCore。",
             "MLP中间量化由VM原生CPU算子执行。该桥接不是通用RunCodegen外部module插件。", "",
             "每个VM/artifact复用32组输入，对照PyTorch与原C++oracle；另外检查非法输入、VM callback异常及错误恢复。", "",
             "| 算例 | VM device calls | core cycles | A/W/Y payload B | 估计总延迟 µs | core时间占比 | 实测VM中位数 µs |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        baseline = row["baseline_estimate"]
        count = baseline["counts"]
        payload = count["load_a_bytes"]+count["load_w_bytes"]+count["store_y_bytes"]
        lines.append("| {} | {} | {} | {} | {:.3f} | {:.1%} | {:.3f} |".format(row["experiment"],
            row["vm_offload_partitions"], count["core_cycles"], payload, baseline["total_ns"]/1000,
            baseline["core_fraction"], row["software_timing"]["vm_execute_ns"]["median_ns"]/1000))
    lines += ["", "估计列采用假定100MHz、TX/RX各10MB/s、8B设备命令及配置中的host成本；实际VM列含CPU模拟/FFI开销，二者不能互换。",
              "所有场景假定顺序执行、每次子程序重新装载权重，没有传输/计算重叠。program仍驻留host，不占64B数据scratchpad。",
              "", "## 频率与链路灵敏度（估计总延迟 µs）", "",
              "| 算例 | 100MHz / 1MB/s | 100MHz / 10MB/s | 100MHz / 50MB/s | 200MHz / 10MB/s |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        estimates = json.loads((output/row["experiment"]/"performance_estimates.json").read_text())
        def latency(hz, rate):
            return next(e["total_ns"]/1000 for e in estimates if e["scenario"]["core_hz"] == hz
                        and e["scenario"]["tx_bytes_per_second"] == rate)
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(row["experiment"],
            latency(100_000_000,1_000_000), latency(100_000_000,10_000_000),
            latency(100_000_000,50_000_000), latency(200_000_000,10_000_000)))
    lines += ["", "核心频率提高只缩短core/setup时间；链路速率、命令和host开销需分别优化。此实验不验证物理Fmax、板卡带宽、SGLang或RTL。",
              "", "逐事件估计、实测软件时间、输入输出与错误记录分别位于各算例目录；配置见performance_config.json。"]
    (output/"analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Runtime experiment PASS: {} models, {} runtime inputs, {} negative cases; {}".format(
        len(rows), sum(row["runtime_inputs"] for row in rows), sum(row["negative_cases"] for row in rows), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
