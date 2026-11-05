#!/usr/bin/env python3
"""Validate common operator models and report their explicit resource costs."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from analyze_esl import load_profiles


def integer_output_shape(op, s):
    if op in ("dot", "reduce_sum", "reduce_max"): return [1]
    if op == "gemv": return [s[0]]
    if op == "gemm": return [s[0], s[2]]
    if op == "conv1d": return [s[0]-s[3]+1, s[2]]
    if op == "conv2d": return [s[0]-s[4]+1, s[1]-s[5]+1, s[3]]
    if op == "dwconv2d": return [s[0]-s[3]+1, s[1]-s[4]+1, s[2]]
    if op == "fir": return [s[0]-s[1]+1]
    if op == "maxpool2d": return [(s[0]-s[3])//s[5]+1, (s[1]-s[4])//s[6]+1, s[2]]
    if op == "transpose": return [s[1], s[0]]
    return list(s)


def integer_oracle(op, shape, a, b):
    if op == "dot":
        return [sum(x*y for x, y in zip(a, b))]
    if op == "gemv":
        m, k = shape
        return [sum(a[t] * b[r*k+t] for t in range(k)) for r in range(m)]
    if op == "gemm":
        m, k, n = shape
        return [sum(a[r*k+t]*b[t*n+c] for t in range(k)) for r in range(m) for c in range(n)]
    if op == "conv1d":
        length, ci, co, kernel = shape
        return [sum(a[(p+t)*ci+i]*b[(t*ci+i)*co+c] for t in range(kernel) for i in range(ci))
                for p in range(length-kernel+1) for c in range(co)]
    if op == "conv2d":
        h, w, ci, co, kh, kw = shape
        return [sum(a[((y+dy)*w+x+dx)*ci+i]*b[((dy*kw+dx)*ci+i)*co+c]
                    for dy in range(kh) for dx in range(kw) for i in range(ci))
                for y in range(h-kh+1) for x in range(w-kw+1) for c in range(co)]
    if op == "dwconv2d":
        h, w, channels, kh, kw = shape
        return [sum(a[((y+dy)*w+x+dx)*channels+c]*b[(dy*kw+dx)*channels+c]
                    for dy in range(kh) for dx in range(kw))
                for y in range(h-kh+1) for x in range(w-kw+1) for c in range(channels)]
    if op == "fir":
        length, k = shape
        return [sum(a[t+k-1-i]*b[i] for i in range(k)) for t in range(length-k+1)]
    if op == "add": return [x+y for x, y in zip(a, b)]
    if op == "mul": return [x*y for x, y in zip(a, b)]
    if op == "relu": return [max(0, x) for x in a]
    if op == "requant":
        return [min(7, max(-8, ((abs(3*x)+2)//4)*(1 if x >= 0 else -1))) for x in a]
    if op == "reduce_sum": return [sum(a)]
    if op == "reduce_max": return [max(a)]
    if op == "maxpool2d":
        h, w, c, kh, kw, sh, sw = shape
        return [max(a[((y*sh+dy)*w+x*sw+dx)*c+i] for dy in range(kh) for dx in range(kw))
                for y in range((h-kh)//sh+1) for x in range((w-kw)//sw+1) for i in range(c)]
    if op == "transpose":
        h, w = shape
        return [a[y*w+x] for x in range(w) for y in range(h)]
    raise ValueError("Unknown integer op " + op)


def nonlinear_oracle(op, shape, x, gamma, beta, epsilon):
    rows, cols = shape
    result = []
    for r in range(rows):
        v = x[r*cols:(r+1)*cols]
        if op == "softmax":
            e = [math.exp(z-max(v)) for z in v]
            result.extend(z/sum(e) for z in e)
        elif op in ("layernorm", "rmsnorm"):
            mean = sum(v)/cols if op == "layernorm" else 0
            denominator = math.sqrt(sum((z-mean)**2 for z in v)/cols+epsilon)
            result.extend((z-mean)/denominator*gamma[i]+(beta[i] if op == "layernorm" else 0)
                          for i, z in enumerate(v))
        elif op == "gelu": result.extend(0.5*z*(1+math.erf(z/math.sqrt(2))) for z in v)
        elif op == "tanh": result.extend(math.tanh(z) for z in v)
        elif op in ("sigmoid", "silu"):
            for z in v:
                sig = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
                result.append(z*sig if op == "silu" else sig)
        else: raise ValueError("Unknown nonlinear op " + op)
    return result


def run_model(command, trace):
    trace.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command+["--trace", str(trace)], capture_output=True, text=True,
                               encoding="utf-8", timeout=60)
    if completed.returncode:
        raise ValueError("Operator CLI failed: " + completed.stderr[:1500])
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or type(result.get("cycles")) is not int or result["cycles"] < 1:
        raise ValueError("Invalid operator result/cycle count")
    if not trace.is_file() or not trace.stat().st_size:
        raise ValueError("Missing operator trace")
    return result


def validate_integer(sample, case, resource):
    if sample.get("op") != case["op"] or sample.get("shape") != case["shape"]:
        raise ValueError("Operator/shape mismatch")
    if sample.get("output_shape") != integer_output_shape(case["op"], case["shape"]):
        raise ValueError("Output shape mismatch")
    expected = integer_oracle(case["op"], case["shape"], sample["a"], sample["b"])
    if sample.get("golden") != expected or sample.get("cycle_result") != expected:
        raise ValueError("Integer operator differs from independent Python oracle: " + case["op"])
    op, s = case["op"], case["shape"]
    terms = {"dot":s[0], "gemv":s[-1], "gemm":s[1] if len(s)>1 else 0,
             "conv1d":s[1]*s[3] if len(s)==4 else 0,
             "conv2d":s[2]*s[4]*s[5] if len(s)==6 else 0,
             "dwconv2d":s[3]*s[4] if len(s)==5 else 0,
             "fir":s[1] if len(s)==2 else 0}
    macs = len(expected)*terms[op] if op in terms else 0
    scalar = (s[0] if op in ("add","mul","relu","requant") else
              s[0]-1 if op in ("reduce_sum","reduce_max") else
              len(expected)*(s[3]*s[4]-1) if op == "maxpool2d" else 0)
    out16 = op in ("dot","gemv","gemm","conv1d","conv2d","dwconv2d","fir","add","mul","reduce_sum")
    counts = {"macs":macs, "scalar_ops":scalar,
              "input_bytes":2*len(sample["a"]) if op == "requant" else (len(sample["a"])+1)//2,
              "weight_bytes":(len(sample["b"])+1)//2,
              "output_bytes":2*len(expected) if out16 else (len(expected)+1)//2}
    for field, count in counts.items():
        if sample.get(field) != count: raise ValueError("{} {} count mismatch".format(op, field))
    for field in ("lanes", "depth", "a_ports", "b_ports"):
        if sample.get(field) != resource[field]: raise ValueError("Resource mismatch")
    for prefix in ("a", "b"):
        if sample["peak_"+prefix+"_reads"] > resource[prefix+"_ports"]:
            raise ValueError("Exceeded operand read-port budget")
    if sample.get("model_kind") != "integer_bit_exact": raise ValueError("Integer fidelity missing")
    return expected


def validate_nonlinear(sample):
    expected = nonlinear_oracle(sample["op"], sample["shape"], sample["x"], sample["gamma"], sample["beta"], sample["epsilon"])
    conditioning = (8*sys.float_info.epsilon*max(map(abs, sample["x"]))*max(map(abs, sample["gamma"]))/math.sqrt(sample["epsilon"])
                    if sample["op"] == "layernorm" else 0)
    for field in ("golden", "cycle_result"):
        if len(sample[field]) != len(expected) or any(not math.isfinite(v) or abs(v-e) > 1e-9*(1+abs(e))+conditioning
                                                   for v, e in zip(sample[field], expected)):
            raise ValueError("Nonlinear result mismatch: " + sample["op"])
    if sample.get("hardware_precision", "missing") is not None:
        raise ValueError("Nonlinear reference must not claim hardware precision")
    return expected


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def report(rows, output):
    lines = ["# Tiny-LPU common operator models", "",
             "Fifteen integer operators use independent bit-exact references and port-constrained cycle execution. Seven nonlinear operators use binary64 mathematics and an independent primitive DAG. Cycles are ESL assumptions, not physical Fmax.", "",
             "## Integer operators: two lanes, depth three, one A port and two B ports", "",
             "Scenarios use 100MHz and a 25MB/s half-duplex host, serialized load/compute/readback, one START byte and the historical sideband-address assumption.", "",
             "| Operator | Shape | Core clocks | MACs | Scalar ops | Resident-data lower bound B | Core us | Sequential total us |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["template"] == "sky" and r["resource"] == "p2_a1" and r["core_mhz"] == 100 and r["io_MBps"] == 25:
            lines.append("| {op} | {shape} | {cycles} | {macs} | {scalar_ops} | {resident_data_bytes} | {core_us:.3f} | {sequential_total_us:.3f} |".format(**r))
    lines.extend(["", "Resident bytes exclude control, addressing, pipelines and register ports; they are not tile-fit evidence. weight_bytes names the second operand even for elementwise data.", "",
                  "## Read ports and mapping", "",
                  "Compare one/four A ports with four lanes, depth three and four B ports. Additional ports have implementation cost.", "",
                  "| Operator | A1 clocks | A4 clocks | Core speedup |",
                  "| --- | ---: | ---: | ---: |"])
    for op in ("gemv", "gemm", "conv2d", "dwconv2d", "fir", "add", "transpose"):
        found = {r["resource"]: r for r in rows if r["template"] == "sky" and r["op"] == op
                 and r["core_mhz"] == 100 and r["io_MBps"] == 25 and r["resource"] in ("p4_a1", "p4_a4")}
        if len(found) == 2:
            lo, hi = found["p4_a1"]["cycles"], found["p4_a4"]["cycles"]
            lines.append("| {} | {} | {} | {:.2f}× |".format(op, lo, hi, lo/hi))
    lines.extend(["", "## Nonlinear 2x8 reference: ALU II=1 and one nonpipelined SFU", "",
                  "ALU latency is one; exp/tanh eight; div/sqrt/erf sixteen clocks. One SFU operation is active at a time; ALU can overlap. Double SFU latency for sensitivity.", "",
                  "| Operator | Default clocks | Double-SFU-latency clocks | Default SFU busy clocks | Primitive counts |",
                  "| --- | ---: | ---: | ---: | --- |"])
    for op in ("softmax", "layernorm", "rmsnorm", "gelu", "silu", "sigmoid", "tanh"):
        found = {r["sfu_scale"]: r for r in rows if r["template"] == "sky" and r["op"] == op
                 and r["core_mhz"] == 100 and r["shape"] == [2, 8]}
        lines.append("| {} | {} | {} | {} | {} |".format(op, found[1]["cycles"], found[2]["cycles"],
                     found[1]["sfu_busy_cycles"], json.dumps(found[1]["primitive_counts"])))
    lines.extend(["", "Double precision is a mathematical reference, not proposed FP64 hardware. Hardware precision, LUT approximations and supply ports are unselected; physical bytes, host latency and INT4 MAC throughput are intentionally unspecified.", "",
                  "## Microarchitecture implications", "",
                  "- GEMV, some GEMM and ordinary convolution can share A per tick and fit output-row parallelism.",
                  "- Depthwise convolution, adjacent FIR outputs and elementwise operations often need distinct A addresses. Compare banking, K parallelism or sliding-window support before adding lanes.",
                  "- ReLU, comparison, reduction, transpose and requant need explicit ALU, movement or quantization units; do not count them all as MAC/s.",
                  "- Softmax and normalization also require reductions, exp/sqrt/division. These models do not imply native nonlinear INT4-PE support.",
                  "- All three templates save results; equal ESL resources give equal clocks. Physical cross-process PPA needs separate evidence.", "",
                  "[Full CSV](summary.csv) - [Raw model results](model_results.json) - [Run configuration](run.json)", ""])
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-model", required=True, type=Path)
    parser.add_argument("--nonlinear-model", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "reports/operators"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs/operators.json").read_text(encoding="utf-8"))
    samples, base = [], []
    for case in config["integer_cases"]:
        for resource in config["integer_resources"]:
            key = case["op"] + "_" + resource["id"]
            command = [str(args.integer_model.resolve()), "--op", case["op"], "--shape", ",".join(map(str, case["shape"])),
                       "--lanes", str(resource["lanes"]), "--depth", str(resource["depth"]),
                       "--a-ports", str(resource["a_ports"]), "--b-ports", str(resource["b_ports"]), "--seed", "1"]
            sample = run_model(command, output / "traces" / (key+".csv"))
            validate_integer(sample, case, resource)
            samples.append({"key": key, "result": sample})
            for frequency in config["core_frequency_mhz"]:
                for bandwidth in config["host_payload_MBps"]:
                    wire_bytes = sample["input_bytes"]+sample["weight_bytes"]+sample["output_bytes"]+1
                    base.append(dict(op=sample["op"], shape=sample["shape"], model_kind=sample["model_kind"],
                                     resource=resource["id"], lanes=resource["lanes"], a_ports=resource["a_ports"], b_ports=resource["b_ports"],
                                     cycles=sample["cycles"], macs=sample["macs"], scalar_ops=sample["scalar_ops"], core_mhz=frequency,
                                     io_MBps=bandwidth, core_us=sample["cycles"]/frequency,
                                     resident_data_bytes=wire_bytes-1, sequential_total_us=wire_bytes/bandwidth+sample["cycles"]/frequency,
                                     peak_a_reads=sample["peak_a_reads"], peak_b_reads=sample["peak_b_reads"], required_units=sample["required_units"]))
    for op in config["nonlinear_ops"]:
        for cols in config["nonlinear_cols"]:
            for scale in config["sfu_scale"]:
                key = "{}_{}x{}_sfu{}".format(op, config["nonlinear_rows"], cols, scale)
                command = [str(args.nonlinear_model.resolve()), "--op", op, "--rows", str(config["nonlinear_rows"]),
                           "--cols", str(cols), "--seed", "1", "--sfu-scale", str(scale)]
                sample = run_model(command, output / "traces" / (key+".csv"))
                validate_nonlinear(sample)
                samples.append({"key": key, "result": sample})
                for frequency in config["core_frequency_mhz"]:
                    base.append(dict(op=op, shape=sample["shape"], model_kind=sample["model_kind"], resource="alu_sfu",
                                     cycles=sample["cycles"], core_mhz=frequency, core_us=sample["cycles"]/frequency, sfu_scale=scale,
                                     sfu_busy_cycles=sample["sfu_busy_cycles"], primitive_counts=sample["primitive_counts"],
                                     required_units=sample["required_units"], hardware_precision=None, resident_data_bytes=None,
                                     sequential_total_us=None))
    rows = []
    for profile in load_profiles(root):
        specific = [dict(template=profile["id"], pdk=profile["pdk"], physical_performance_validated=False, **r) for r in base]
        folder = output / profile["id"]
        folder.mkdir(exist_ok=True)
        write_csv(folder / "performance.csv", specific)
        write_json(folder / "profile.json", profile)
        rows.extend(specific)
    write_csv(output / "summary.csv", rows)
    write_json(output / "model_results.json", samples)
    write_json(output / "run.json", {"config": config, "rows": len(rows), "samples": len(samples),
                                     "integer_model": str(args.integer_model.resolve()), "nonlinear_model": str(args.nonlinear_model.resolve()),
                                     "physical_performance_validated": False})
    report(rows, output)
    print("Operators passed: {} operators, {} validated model runs, {} report rows; {}".format(
        len(config["integer_cases"])+len(config["nonlinear_ops"]), len(samples), len(rows), output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
        print("Operator analysis failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
