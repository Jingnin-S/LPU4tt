#!/usr/bin/env python3
"""Run real export/Relax partition/codegen -> independent C++ plan experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))
from tinylpu_compiler import Target, compile_plan
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.plan import verify_artifact


def run_json(command):
    run = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError("Command failed: {}\n{}".format(command[0], run.stderr[:2000]))
    return json.loads(run.stdout)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def build_models(torch):
    def weight(rows, k, phase):
        return [[(r*5+c*3+phase+r)%5-2 for c in range(k)] for r in range(rows)]

    class Linear(torch.nn.Module):
        def __init__(self, rows, k):
            super().__init__()
            self.register_buffer("w", torch.tensor(weight(rows,k,1), dtype=torch.float32))
        def forward(self, x):
            return torch.nn.functional.linear(x, self.w)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w1", torch.tensor(weight(4,16,1), dtype=torch.float32))
            self.register_buffer("w2", torch.tensor(weight(4,4,3), dtype=torch.float32))
        def forward(self, x):
            h = torch.nn.functional.linear(x, self.w1)
            h = torch.clamp(torch.floor(torch.relu(h)*0.75+0.5), 0, 7)
            return torch.nn.functional.linear(h, self.w2)

    return [("linear_16x8",Linear(8,16),16), ("linear_tail_16x7",Linear(7,16),16),
            ("linear_odd_k17x3",Linear(3,17),17), ("mlp_16x4x4",Mlp(),16)]


def model_oracle(oracle, model, values):
    def execute(op, shape, a, b=None):
        command = [str(oracle), "--op", op, "--shape", ",".join(map(str,shape)),
                   "--lanes", "2", "--depth", "3", "--a-ports", "1", "--b-ports", "2", "--seed", "1",
                   "--a", ",".join(map(str,a))]
        if b is not None: command += ["--b", ",".join(str(int(v)) for row in b for v in row)]
        result = run_json(command)
        if result["golden"] != result["cycle_result"]: raise RuntimeError("Existing model self-check failed")
        return result["golden"]
    if hasattr(model, "w"):
        w = model.w.to(dtype=__import__("torch").int32).tolist()
        return execute("gemv", [len(w),len(values)], values, w)
    w1, w2 = model.w1.tolist(), model.w2.tolist()
    first = execute("gemv", [4,16], values, w1)
    quantized = execute("requant", [4], [max(0,v) for v in first])
    return execute("gemv", [4,4], quantized, w2)


def negative_cases(sim, folder, input_values):
    program, weights = folder / "program.tlp", folder / "weights.bin"
    source = program.read_bytes()
    records = []
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        cases = {"bad_magic":b"BADMAGIC"+source[8:], "bad_abi":source[:8]+struct.pack("<I",99)+source[12:],
                 "truncated_program":source[:-1]}
        for name, content in cases.items():
            path = temp / (name+".tlp")
            path.write_bytes(content)
            run = subprocess.run([str(sim),"--program",str(path),"--weights",str(weights),"--input",",".join(map(str,input_values))],
                                 text=True,capture_output=True,timeout=30)
            if run.returncode == 0: raise RuntimeError("Malformed program accepted: "+name)
            records.append({"case":name,"exit_code":run.returncode,"error":run.stderr.strip()})
        for name, values in (("input_out_of_range",[8]*len(input_values)), ("input_wrong_length",input_values[:-1])):
            run = subprocess.run([str(sim),"--program",str(program),"--weights",str(weights),"--input",",".join(map(str,values))],
                                 text=True,capture_output=True,timeout=30)
            if run.returncode == 0: raise RuntimeError("Invalid input accepted: "+name)
            records.append({"case":name,"exit_code":run.returncode,"error":run.stderr.strip()})
        bad_weight = temp / "short_weights.bin"
        bad_weight.write_bytes(weights.read_bytes()[:-1])
        run = subprocess.run([str(sim),"--program",str(program),"--weights",str(bad_weight),"--input",",".join(map(str,input_values))],
                             text=True,capture_output=True,timeout=30)
        if run.returncode == 0: raise RuntimeError("Truncated weights accepted")
        records.append({"case":"truncated_weights","exit_code":run.returncode,"error":run.stderr.strip()})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/compiler")
    parser.add_argument("--target", type=Path, default=ROOT / "configs/compiler_target.json")
    args = parser.parse_args(argv)
    import torch
    import tvm

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = Target(**json.loads(args.target.read_text(encoding="utf-8")))
    rows, details = [], []
    rng = random.Random(53)
    for name, model, k in build_models(torch):
        folder = output / name
        start = time.perf_counter()
        node, metadata = import_model(model, (torch.zeros(1,k,dtype=torch.float32),), folder)
        manifest = compile_plan(node, folder, target, name)
        compile_seconds = time.perf_counter()-start
        write_json(folder / "frontend.json", metadata)
        inputs = [[0]*k,[-8]*k,[7]*k,[-8 if i%2 else 7 for i in range(k)]]
        inputs.extend([rng.choices(range(-8,8),k=k) for _ in range(28)])
        samples = []
        program_hash = manifest["program_sha256"]
        for index, values in enumerate(inputs):
            verify_artifact(folder)
            command = [str(args.sim.resolve()),"--program",str(folder/"program.tlp"),"--weights",str(folder/"weights.bin"),
                       "--input",",".join(map(str,values))]
            if index == 0: command += ["--trace",str(folder/"execution_trace.csv")]
            actual = run_json(command)
            with torch.no_grad():
                torch_values = model(torch.tensor([values],dtype=torch.float32)).reshape(-1).tolist()
            if any(v != round(v) for v in torch_values): raise RuntimeError("Reference left integer subset")
            expected = [int(v) for v in torch_values]
            existing = model_oracle(args.oracle.resolve(), model, values)
            if actual["result"] != expected or actual["result"] != existing:
                raise RuntimeError("Compiled plan numerical mismatch for {} input {}".format(name,index))
            if actual["core_cycles"] != manifest["predicted_core_cycles"] or actual["kernel_count"] != len(manifest["kernels"]):
                raise RuntimeError("Runtime core timing differs from compiler model")
            if actual["macs"] != manifest["predicted_macs"]:
                raise RuntimeError("Runtime MAC count differs from compiler model")
            for field, count in manifest["predicted_transfers"].items():
                if actual[field] != count: raise RuntimeError("Runtime byte accounting mismatch")
            if actual["host_ops"] != manifest["predicted_host_ops"]: raise RuntimeError("Host-step count mismatch")
            samples.append({"input":values,"torch_result":expected,"existing_model_result":existing,"runtime":actual})
        if len({tuple(s["runtime"]["result"]) for s in samples}) < 2:
            raise RuntimeError("Input-variation test did not produce distinct outputs")
        if verify_artifact(folder)["program_sha256"] != program_hash: raise RuntimeError("Program changed with runtime input")
        negatives = negative_cases(args.sim.resolve(),folder,inputs[0])
        write_json(folder/"results.json",{"samples":samples,"negative_cases":negatives,"same_artifact_all_inputs":True})
        row = {"experiment":name,"runtime_inputs":len(samples),"kernels":len(manifest["kernels"]),
               "byoc_partitions":len(metadata["byoc_functions"]),"core_cycles":manifest["predicted_core_cycles"],
               "program_bytes":manifest["program_bytes"],"weights_bytes":manifest["weights_bytes"],
               "load_a_bytes":manifest["predicted_transfers"]["load_a_bytes"],
               "load_w_bytes":manifest["predicted_transfers"]["load_w_bytes"],
               "store_y_bytes":manifest["predicted_transfers"]["store_y_bytes"],
               "host_ops":manifest["predicted_host_ops"],"negative_cases":len(negatives),
               "compile_wall_seconds":compile_seconds,"all_numerical_checks_passed":True}
        rows.append(row)
        details.append({"summary":row,"frontend":metadata,"manifest":manifest})
    write_json(output/"results.json",{"experiments":details,"torch_version":torch.__version__,"tvm_version":tvm.__version__,
                                      "sim":str(args.sim.resolve()),"oracle":str(args.oracle.resolve()),
                                      "input_count":sum(r["runtime_inputs"] for r in rows),"physical_performance_validated":False})
    usage = {}
    for experiment in details:
        for instruction in experiment["manifest"]["instructions"]:
            entry = usage.setdefault(instruction["op"], {"count":0,"max_unsigned_args":[0]*7})
            entry["count"] += 1
            entry["max_unsigned_args"] = [max(a,b) for a,b in zip(entry["max_unsigned_args"],instruction["args"])]
    write_json(output/"isa_usage.json", {"status":"experimental command ABI, not a frozen RTL ISA", "operations":usage})
    lines = ["# 编译到独立模拟器：首轮实验", "", "真实执行链：torch.export → TVM Relax导入/BYOC分区 → Tiny-LPU IR/分块/地址规划 → binary plan → C++解释器/DseCore。",
             "", "C++解释器在运行时从scratchpad读取操作数，未使用编译时预计算输入乘积。MLP的ReLU/3-4 requant是明确host步骤。", "",
             "| 实验 | 新输入组 | BYOC分区 | GEMV tiles | core cycles | program B | weights B | A装载 B | host量化元素 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| {experiment} | {runtime_inputs} | {byoc_partitions} | {kernels} | {core_cycles} | {program_bytes} | {weights_bytes} | {load_a_bytes} | {host_ops} |".format(**r))
    lines.extend(["", "每份artifact跑32组输入，含零、正负极值与随机输入；结果同时与PyTorch模型及既有C++算子oracle一致。各实验另验证坏magic、ABI、截断program/weights及非法输入被拒绝。", "",
                  "程序是host执行的实验命令ABI，program字节不占据64B数据scratchpad；不是已冻结的RTL ISA。buffers预留两个槽，但本轮为同步顺序执行，不宣称已经重叠传输。", "",
                  "本轮真实完成TVM导入与BYOC分区，Tiny后端读取实际分区后IR并输出独立程序。TVM VM外部runtime module链接、SGLang适配和实际硬件尚未接通。100MHz仍是分析目标，没有实测Fmax/芯片吞吐结论。", "",
                  "[结构化结果](results.json) · [候选命令使用统计](isa_usage.json)。每个实验目录含imported/partitioned Relax IR、Tiny IR、manifest、binary program、weights与执行trace。", ""])
    (output/"analysis.md").write_text("\n".join(lines),encoding="utf-8")
    print("Compiler experiment PASS: {} models, {} runtime inputs, {} negative cases; {}".format(
        len(rows),sum(r["runtime_inputs"] for r in rows),sum(r["negative_cases"] for r in rows),output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print("Compiler experiment failed: {}".format(error),file=sys.stderr)
        sys.exit(1)
