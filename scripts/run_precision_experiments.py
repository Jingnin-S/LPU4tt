#!/usr/bin/env python3
"""Integer-first numerical comparisons; additional formats remain host references."""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"compiler"))
from tinylpu_compiler import Target
from tinylpu_compiler.batch_plan import compile_batch_linear
from tinylpu_compiler.precision import (BITS, MODES, NumericMode, PrecisionRuntime, assert_same_float32,
    error_metrics, ordered_reference, quantization_scale, quantize, required_accumulator_bits,
    storage_requirement, torch_convert)
from tinylpu_compiler.runtime import PlanSession


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def array_hash(values):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def fixture(spec, distribution, seed, frames, calibration_frames):
    import numpy as np
    rng = np.random.default_rng(seed)
    k = spec["layers"][0][1]
    weights = []
    for rows, columns in spec["layers"]:
        if distribution == "cancellation":
            signs = np.where(np.arange(columns)%2, -1.0, 1.0)
            weight = np.stack([signs*(row+1)/rows for row in range(rows)])
        elif distribution == "wide_dynamic":
            weight = rng.choice([-1,1], (rows,columns))*10**rng.uniform(-6,2,(rows,columns))
        else:
            weight = rng.normal(0,1/(columns**0.5),(rows,columns))
        weights.append(np.asarray(weight, dtype="float32"))

    def inputs(count, calibration):
        if distribution == "small":
            return rng.normal(0,1e-9,(count,k))
        if distribution == "large":
            return rng.normal(0,1e5,(count,k))
        if distribution == "wide_dynamic":
            return rng.choice([-1,1],(count,k))*10**rng.uniform(-6,4,(count,k))
        if distribution == "outlier_shift":
            result = rng.normal(0,.1,(count,k))
            if not calibration:
                for frame in range(count):
                    result[frame,frame%k] = 1000 if frame%2 else -1000
            return result
        if distribution == "cancellation":
            return 1+rng.uniform(-2e-4,2e-4,(count,k))
        if distribution == "zero":
            return np.zeros((count,k))
        return rng.normal(0,1,(count,k))

    return (np.asarray(inputs(frames,False),dtype="float32"),
            np.asarray(inputs(calibration_frames,True),dtype="float32"), weights)


def torch_model(values, weights, dtype):
    import numpy as np
    import torch
    current = torch.from_numpy(np.asarray(values)).to(dtype=dtype)
    intermediates = []
    with torch.no_grad():
        for index, weight in enumerate(weights):
            intermediates.append(current.numpy().copy())
            current = torch.nn.functional.linear(current,torch.from_numpy(weight).to(dtype=dtype))
            if index+1 < len(weights):
                current = torch.relu(current)
    return current.numpy().copy(), intermediates


def conversion(runtime, values, dtype):
    import numpy as np
    if dtype == "float32":
        return np.asarray(values,dtype="float32"), {"conversion_overflows":0,"conversion_zero_losses":0}
    decoded, encoded = runtime.convert(values,dtype)
    expected, expected_bits = torch_convert(values,dtype)
    assert_same_float32(decoded,expected,"C++/Torch input/output conversion")
    finite = ~np.isnan(expected)
    if not np.array_equal(encoded[finite],expected_bits[finite]):
        raise RuntimeError("C++/Torch 16-bit encoding mismatch")
    source = np.asarray(values,dtype="float32")
    return decoded, {"conversion_overflows":int(np.count_nonzero(np.isfinite(source)&~np.isfinite(decoded))),
                     "conversion_zero_losses":int(np.count_nonzero(np.isfinite(source)&(source!=0)&(decoded==0)))}


def run_mode(runtime, values, weights, calibration_activations, mode, policy, folder, target, legacy_library):
    import numpy as np
    actual, expected = values.copy(), values.copy()
    totals = {name:0 for name in ("integer_overflow_events","nonfinite_product_events",
              "nonfinite_accumulator_events","input_conversion_overflows","input_conversion_zero_losses",
              "output_conversion_overflows","output_conversion_zero_losses","activation_clipped_elements",
              "weight_clipped_elements")}
    layers, saved = [], {}
    for layer, weight in enumerate(weights):
        rows, k = weight.shape
        raw_actual, raw_expected, quantized_a = [], [], []
        a_scales, expected_a_scales = [], []
        if mode.activation_dtype.startswith("int"):
            bits = BITS[mode.activation_dtype]
            w_scale = quantization_scale(weight,bits,per_row=True)
            qw, w_clipped = quantize(weight,w_scale,bits,per_row=True)
            totals["weight_clipped_elements"] += w_clipped
            static_scale = quantization_scale(calibration_activations[layer],bits)
            for frame in range(values.shape[0]):
                sa = quantization_scale(actual[frame],bits) if policy == "dynamic_absmax" else static_scale
                se = quantization_scale(expected[frame],bits) if policy == "dynamic_absmax" else static_scale
                qa, clipped = quantize(actual[frame],sa,bits)
                qe, _ = quantize(expected[frame],se,bits)
                totals["activation_clipped_elements"] += clipped
                result, counters = runtime.gemv(qa,qw,mode)
                oracle, expected_counters = ordered_reference(qe,qw,mode)
                if not np.array_equal(result,oracle) or counters != expected_counters:
                    raise RuntimeError("Integer reference/counter mismatch")
                for key,count in counters.items():
                    totals[key] += count
                raw_actual.append(result)
                raw_expected.append(oracle)
                quantized_a.append(qa)
                a_scales.append(float(sa))
                expected_a_scales.append(float(se))
            raw_actual, raw_expected = np.asarray(raw_actual),np.asarray(raw_expected)
            actual = np.asarray(raw_actual*np.asarray(a_scales)[:,None]*w_scale.astype(np.float64)[None,:],dtype="float32")
            expected = np.asarray(raw_expected*np.asarray(expected_a_scales)[:,None]*w_scale.astype(np.float64)[None,:],dtype="float32")
            assert_same_float32(actual,expected,"dequantized integer pipeline")
            layer_info = {"activation_scales":a_scales,"weight_scales":w_scale.tolist(),
                          "zero_point":0,"rounding":"nearest_ties_even","raw_qrange":[-(1<<(bits-1)),(1<<(bits-1))-1],
                          "calibration":"runtime per-frame absmax" if policy == "dynamic_absmax" else "independent FP32 calibration activations"}
            saved["layer{}_qa".format(layer)] = np.asarray(quantized_a,dtype="int32")
            saved["layer{}_qw".format(layer)] = qw
            if mode.name == "int4_acc16" and k <= 64 and rows <= 16:
                artifact = folder/"legacy"/(mode.name+"_"+policy)/("layer{}".format(layer))
                manifest = compile_batch_linear(qw.tolist(),values.shape[0],artifact,target,
                                    "precision_legacy",order="weight_stationary",weight_policy="resident")
                with PlanSession(artifact,legacy_library) as session:
                    report = session.run(np.asarray(quantized_a,dtype="float32").reshape(1,-1))
                if report["result"] != raw_actual.reshape(-1).tolist():
                    raise RuntimeError("ABI-v1 integer path differs from numeric reference")
                for key,value in {"core_cycles":manifest["predicted_core_cycles"], **manifest["predicted_transfers"]}.items():
                    if report[key] != value:
                        raise RuntimeError("Legacy integer byte/cycle accounting mismatch")
                write_json(artifact/"validation.json",{"passed":True,"raw_result_sha256":array_hash(raw_actual),
                           "runtime":report,"input":np.asarray(quantized_a).tolist()})
                layer_info["legacy_plan_verified"] = True
            else:
                layer_info["legacy_plan_verified"] = False
                layer_info["legacy_reason"] = "format/accumulator not ABI-v1 or K exceeds64"
        else:
            layer_info = {"rounding":"nearest_ties_even","subnormals":"preserved",
                          "accumulation":"ordered FP32 multiply then add, no FMA"}
            _, stats = conversion(runtime,actual,mode.activation_dtype)
            _, weight_stats = conversion(runtime,weight,mode.weight_dtype)
            for name in ("conversion_overflows","conversion_zero_losses"):
                totals["input_"+name] += stats[name]+weight_stats[name]
            for a,e in zip(actual,expected):
                result,counters = runtime.gemv(a,weight,mode)
                oracle,expected_counters = ordered_reference(e,weight,mode)
                assert_same_float32(result,oracle,"ordered floating GEMV")
                if counters != expected_counters:
                    raise RuntimeError("Floating diagnostic counter mismatch")
                for key,count in counters.items():
                    totals[key] += count
                raw_actual.append(result)
                raw_expected.append(oracle)
            raw_actual, raw_expected = np.asarray(raw_actual),np.asarray(raw_expected)
            actual,stats = conversion(runtime,raw_actual.astype("float32"),mode.output_dtype)
            expected,_ = torch_convert(raw_expected.astype("float32"),mode.output_dtype)
            assert_same_float32(actual,expected,"floating output storage conversion")
            for name in ("conversion_overflows","conversion_zero_losses"):
                totals["output_"+name] += stats[name]
        saved["layer{}_raw".format(layer)] = raw_actual
        layer_info["shape"] = [rows,k]
        layer_info["storage"] = storage_requirement(rows,k,mode,quantization=policy)
        layers.append(layer_info)
        if layer+1 < len(weights):
            actual,expected = np.maximum(actual,0),np.maximum(expected,0)
    saved["output"] = actual
    return actual,totals,layers,saved


def accumulator_diagnostics(runtime):
    import numpy as np
    records = []
    for dtype,bits,k in (("int4",16,64),("int4",16,256),("int4",16,512),("int4",32,512),
                         ("int8",16,2),("int8",32,64),("int8",32,4096)):
        operand = -(1 << (BITS[dtype]-1))
        mode = NumericMode("raw_range",dtype,dtype,"int{}".format(bits),"int{}".format(bits))
        a,w = np.full(k,operand,dtype="float32"),np.full((1,k),operand,dtype="float32")
        result,counters = runtime.gemv(a,w,mode)
        oracle,expected = ordered_reference(a,w,mode)
        if not np.array_equal(result,oracle) or counters != expected:
            raise RuntimeError("Accumulator range diagnostic mismatch")
        records.append({"operand_dtype":dtype,"accumulator_bits":bits,"k":k,"result":int(result[0]),
                        "mathematical_result":k*operand*operand,**counters,
                        **required_accumulator_bits(BITS[dtype],BITS[dtype],k)})
    return records


def main(argv=None):
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-library",required=True,type=Path)
    parser.add_argument("--legacy-library",required=True,type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/precision_experiments.json")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/precision")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    folder = args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/"config_snapshot.json",config)
    runtime = PrecisionRuntime(args.numeric_library)
    target = Target(**config["integer_plan_validation_target"])
    rows,case_summaries = [],[]
    for case_index,spec in enumerate(config["cases"]):
        for dist_index,distribution in enumerate(config["distributions"]):
            name = spec["name"]+"__"+distribution
            output_dir = folder/name
            output_dir.mkdir(parents=True,exist_ok=True)
            values,calibration,weights = fixture(spec,distribution,config["seed"]+case_index*1000+dist_index,
                                                config["batch_frames"],config["calibration_frames"])
            reference32,_ = torch_model(values,weights,torch.float32)
            reference64,_ = torch_model(values.astype("float64"),weights,torch.float64)
            _,calibration_activations = torch_model(calibration,weights,torch.float32)
            if not np.all(np.isfinite(reference32)):
                raise RuntimeError("Fixture FP32 baseline must remain finite")
            np.savez_compressed(output_dir/"fixture.npz",input=values,calibration=calibration,
                                reference_fp32=reference32,reference_fp64=reference64,
                                **{"weight{}".format(i):w for i,w in enumerate(weights)})
            results,outputs = [],{}
            for mode in MODES:
                policies = config["activation_quantization"] if mode.activation_dtype.startswith("int") else ["none"]
                for policy in policies:
                    actual,counters,layers,saved = run_mode(runtime,values,weights,calibration_activations,
                                                          mode,policy,output_dir,target,args.legacy_library)
                    metric = error_metrics(reference32,actual)
                    result = {"case":spec["name"],"distribution":distribution,"mode":mode.to_dict(),
                              "quantization":policy,"input_sha256":array_hash(values),
                              "weight_sha256":[array_hash(w) for w in weights],"metrics_vs_pytorch_fp32":metric,
                              "metrics_vs_fp64":error_metrics(reference64,actual),"counters":counters,
                              "layers":layers,"numeric_oracle_verified":True}
                    results.append(result)
                    rows.append({"case":spec["name"],"distribution":distribution,"mode":mode.name,
                                 "quantization":policy,"input_sha256":array_hash(values),
                                 **{k:v for k,v in metric.items() if k != "finite_subset"},**counters,
                                 "legacy_layers_verified":sum(bool(layer.get("legacy_plan_verified")) for layer in layers)})
                    for key,value in saved.items():
                        outputs[mode.name+"__"+policy+"__"+key] = value
            write_json(output_dir/"results.json",{"reference":"PyTorch FP32 model; FP64 diagnostics also saved",
                       "fp32_vs_fp64":error_metrics(reference64,reference32),"results":results})
            np.savez_compressed(output_dir/"outputs.npz",**outputs)
            case_summaries.append({"case":spec["name"],"distribution":distribution,"result_rows":len(results)})
            print("validated "+name,flush=True)
    diagnostics = accumulator_diagnostics(runtime)
    write_json(folder/"accumulator_diagnostics.json",diagnostics)
    with (folder/"summary.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for mode in MODES:
        policies = config["activation_quantization"] if mode.activation_dtype.startswith("int") else ["none"]
        for policy in policies:
            selected = [r for r in rows if r["mode"] == mode.name and r["quantization"] == policy]
            normal = [r["nrmse"] for r in selected if r["distribution"] == "normal" and r["nrmse"] is not None]
            small = [r["nrmse"] for r in selected if r["distribution"] == "small" and r["nrmse"] is not None]
            groups.append({"mode":mode.name,"quantization":policy,"normal_median_nrmse":statistics.median(normal),
                           "small_median_nrmse":statistics.median(small),
                           "nonfinite_result_rows":sum(not r["complete_finite"] for r in selected),
                           "integer_overflow_events":sum(r["integer_overflow_events"] for r in selected),
                           "activation_clipped_elements":sum(r["activation_clipped_elements"] for r in selected)})
    summary = {"cases":case_summaries,"numeric_rows":len(rows),"summary_groups":groups,
               "legacy_plan_layer_validations":sum(r["legacy_layers_verified"] for r in rows),
               "source_frames_per_case":config["batch_frames"],"torch_version":torch.__version__,
               "numpy_version":np.__version__,"numeric_library":str(args.numeric_library.resolve()),
               "supported_device_abi":"unchanged INT4/ACC16 ABI v1 only",
               "new_formats_are_numeric_references":True,"physical_performance_validated":False}
    write_json(folder/"results.json",summary)
    lines = ["# Integer and mixed-precision numerical experiment", "",
             "INT4/INT8/FP16/BF16 share FP32 source inputs and weights; PyTorch FP32 is the baseline, with FP64 diagnostics.",
             "C++ values/counters are cross-checked against independent Python integer arithmetic or Torch conversion plus sequential NumPy FP32.",
             "FP16/BF16 use RNE inputs, retain subnormals and use separate sequential FP32 multiply/add; OUT16 and OUT32 are separate.", "",
             "{} workload/distribution fixtures, {} mode/quantization rows, {} inputs per fixture, {} historical ABI-v1 layer validations.".format(
                 len(case_summaries),len(rows),config["batch_frames"],summary["legacy_plan_layer_validations"]), "",
             "| Mode | Activation quantization | Normal median NRMSE | Small median NRMSE | Nonfinite-result rows |",
             "| --- | --- | ---: | ---: | ---: |"]
    for group in groups:
        lines.append("| {mode} | {quantization} | {normal_median_nrmse:.6g} | {small_median_nrmse:.6g} | {nonfinite_result_rows} |".format(**group))
    lines += ["", "NRMSE=norm(candidate-reference)/norm(reference). Global error is null for NaN/Inf results; finite subsets are reported separately rather than discarding failures.",
              "", "Dynamic A scale requires a host scan; static A scale uses independent FP32 calibration activations. W scale is offline per-output-channel, zero-point zero, RNE and FP32 scale storage. outlier_shift includes unseen test outliers.",
              "All distributions are synthetic and do not establish LLM perplexity, task accuracy or real-model calibration. The MLP is ordinary Linear-ReLU-Linear, distinct from the fixed-3/4 toy graph.",
              "", "## Integer accumulation range", "",
              "| Operand type | K | Accumulator bits | Mathematical result | Model result | Overflow events |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in diagnostics:
        lines.append("| {operand_dtype} | {k} | {accumulator_bits} | {mathematical_result} | {result} | {integer_overflow_events} |".format(**row))
    lines += ["", "Wrap identifies unsafe historical reference accumulation. K256/1024 here are numerical studies, not new capabilities of ABI-v1. Current ISA-v0.1 tiling is a separate experiment.",
              "", "## Full 8x16 kernel storage budget: two slots, two lanes", "",
              "| Mode | Data scratchpad B | Host quantization parameters B | Accumulator registers B |",
              "| --- | ---: | ---: | ---: |"]
    for mode in MODES:
        storage = storage_requirement(8,16,mode)
        lines.append("| {} | {} | {} | {} |".format(mode.name,storage["data_scratchpad_reserved_bytes"],
            storage["host_quantization_parameter_bytes"],storage["accumulator_register_bytes"]))
    lines += ["", "Budgets include lane padding/output alignment; quantization parameters stay on the host. Control, ports and pipelines are extra. These are not synthesis areas or INT8/FP16/BF16 frequency promises.",
              "Historical bits=16 remains an integer width. Other dtypes exist in explicit NumericMode references; future hardware formats require complete dtype, rounding, scale, FTZ, exception and overflow semantics."]
    (folder/"analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Precision experiment PASS: {} fixtures, {} numeric rows, {} legacy layer validations; {}".format(
          len(case_summaries),len(rows),summary["legacy_plan_layer_validations"],folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
