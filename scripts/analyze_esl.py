#!/usr/bin/env python3
"""Run the Tiny-LPU ESL model and compare template assumptions, without PPA claims."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


TEMPLATES = ("sky", "ihp", "gf")
KERNELS = ("dot4", "gemv2x4")
INPUT_A = (1, -2, 3, -4)
INPUT_W = {"dot4": (7, -8, -1, 2), "gemv2x4": (7, -8, -1, 2, -8, 7, 0, -1)}
PHASES = (
    "weight_load_cycles", "input_load_cycles", "setup_cycles", "compute_cycles",
    "readout_cycles", "host_wait_cycles",
)
PROFILE_FIELDS = (
    "id", "template_repository", "template_commit", "source_urls", "action_ref",
    "pdk", "librelane_version", "clock_period_ns", "density_pct", "tile_width_um",
    "tile_height_um", "tile_die_area_um2", "physical_performance_validated", "assumptions",
)


class AnalysisError(RuntimeError):
    """A model run or input violated the analysis contract."""


def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("{} must be a number".format(name))
    if not math.isfinite(value) or value <= 0:
        raise AnalysisError("{} must be finite and positive".format(name))
    return float(value)


def load_profiles(root: Path) -> List[Dict[str, Any]]:
    profiles = []
    for template in TEMPLATES:
        path = root / "configs" / "templates" / (template + ".json")
        try:
            profile = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise AnalysisError("Cannot load profile {}: {}".format(path, exc)) from exc
        if not isinstance(profile, dict):
            raise AnalysisError("Profile {} must be a JSON object".format(path))
        missing = set(PROFILE_FIELDS) - set(profile)
        if missing:
            raise AnalysisError("Profile {} missing: {}".format(path, ", ".join(sorted(missing))))
        for field in ("id", "template_repository", "template_commit", "action_ref", "pdk", "librelane_version"):
            if not isinstance(profile[field], str) or not profile[field].strip():
                raise AnalysisError("{}.{} must be a nonempty string".format(template, field))
        for field in ("clock_period_ns", "density_pct", "tile_width_um", "tile_height_um", "tile_die_area_um2"):
            positive_number(profile[field], template + "." + field)
        if profile["density_pct"] > 100:
            raise AnalysisError("{}.density_pct exceeds 100".format(template))
        if not math.isclose(profile["tile_width_um"] * profile["tile_height_um"],
                            profile["tile_die_area_um2"], rel_tol=1e-5, abs_tol=0.02):
            raise AnalysisError("{}.tile_die_area_um2 disagrees with dimensions".format(template))
        if profile["physical_performance_validated"] is not False:
            raise AnalysisError("{} must mark physical_performance_validated=false".format(template))
        sources = profile["source_urls"]
        if isinstance(sources, dict):
            if not all(isinstance(key, str) and key for key in sources):
                raise AnalysisError("{}.source_urls keys must be nonempty strings".format(template))
            sources = list(sources.values())
        if not isinstance(sources, list) or not sources or not all(
                isinstance(url, str) and url.startswith("https://") for url in sources):
            raise AnalysisError("{}.source_urls must contain nonempty HTTPS URLs".format(template))
        if not isinstance(profile["assumptions"], list) or not all(
                isinstance(item, str) for item in profile["assumptions"]):
            raise AnalysisError("{}.assumptions must be a string list".format(template))
        profiles.append(dict(profile, analysis_template=template))
    if len({profile["id"] for profile in profiles}) != len(profiles):
        raise AnalysisError("Template profile IDs must be unique")
    return profiles


def expected_counts(kernel: str, reuse: int, host_wait: int) -> Dict[str, int]:
    if kernel not in KERNELS:
        raise AnalysisError("Unsupported kernel: " + str(kernel))
    if type(reuse) is not int or reuse < 1 or type(host_wait) is not int or host_wait < 0:
        raise AnalysisError("reuse must be a positive integer; host_wait must be a nonnegative integer")
    rows = 1 if kernel == "dot4" else 2
    result = {
        "rows": rows, "k": 4, "reuse": reuse, "host_wait": host_wait,
        "transactions": reuse, "macs": 4 * rows * reuse,
        "weight_load_cycles": 2 * rows, "input_load_cycles": 2 * reuse,
        "setup_cycles": reuse, "compute_cycles": 5 * rows * reuse,
        "readout_cycles": 2 * rows * reuse, "host_wait_cycles": host_wait * reuse,
    }
    result["total_cycles"] = sum(result[field] for field in PHASES)
    return result


def golden_result(kernel: str) -> List[int]:
    weights = INPUT_W[kernel]
    return [sum(a * w for a, w in zip(INPUT_A, weights[offset:offset + 4]))
            for offset in range(0, len(weights), 4)]


def validate_model_output(result: Any, kernel: str, reuse: int, host_wait: int) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise AnalysisError("Model stdout must contain one JSON object")
    if result.get("kernel") != kernel:
        raise AnalysisError("Model kernel does not match request")
    for field, expected in expected_counts(kernel, reuse, host_wait).items():
        if type(result.get(field)) is not int or result[field] != expected:
            raise AnalysisError("Model {}: expected {}, got {!r}".format(field, expected, result.get(field)))
    golden = golden_result(kernel)
    for field in ("functional_result", "cycle_result"):
        values = result.get(field)
        if not isinstance(values, list) or any(type(value) is not int for value in values) or values != golden:
            raise AnalysisError("Model {}: expected {}, got {!r}".format(field, golden, values))
    for field, expected in (("peak_macs_per_cycle", 1.0), ("core_macs_per_cycle", 0.8)):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(value, expected):
            raise AnalysisError("Model {} must equal {}".format(field, expected))
    return result


def run_model(model: Path, kernel: str, reuse: int, host_wait: int,
              trace_path: Path = None) -> Dict[str, Any]:
    command = [str(model), "--kernel", kernel,
               "--a", ",".join(map(str, INPUT_A)), "--w", ",".join(map(str, INPUT_W[kernel])),
               "--reuse", str(reuse), "--host-wait", str(host_wait)]
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--trace", str(trace_path)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError("Cannot run model {}: {}".format(model, exc)) from exc
    if completed.returncode != 0:
        raise AnalysisError("Model exited {} for {} reuse={} wait={}: {}".format(
            completed.returncode, kernel, reuse, host_wait,
            (completed.stderr or completed.stdout).strip()[:2000]))
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise AnalysisError("Model stdout is not valid JSON: {}".format(completed.stdout[:300])) from exc
    validate_model_output(result, kernel, reuse, host_wait)
    if trace_path is not None and (not trace_path.is_file() or trace_path.stat().st_size == 0):
        raise AnalysisError("Model did not produce requested trace: {}".format(trace_path))
    return result


def make_metrics(profile: Dict[str, Any], sample: Dict[str, Any], frequency_mhz: float) -> Dict[str, Any]:
    frequency = positive_number(frequency_mhz, "frequency_mhz")
    sample = validate_model_output(sample, sample.get("kernel"), sample.get("reuse"), sample.get("host_wait"))
    total = sample["total_cycles"]
    macs = sample["macs"]
    io_cycles = sum(sample[field] for field in ("weight_load_cycles", "input_load_cycles", "readout_cycles"))
    row = {
        "template": profile["analysis_template"], "profile_id": profile["id"],
        "pdk": profile["pdk"], "action_ref": profile["action_ref"],
        "template_clock_period_ns": profile["clock_period_ns"],
        "template_nominal_frequency_mhz": 1000.0 / profile["clock_period_ns"],
        "assumed_frequency_mhz": frequency,
        "frequency_is_assumption": True, "physical_performance_validated": False,
        "tile_die_area_um2": profile["tile_die_area_um2"],
        "kernel": sample["kernel"], "reuse": sample["reuse"], "host_wait": sample["host_wait"],
        "transactions": sample["transactions"], "macs": macs,
        "functional_result": sample["functional_result"], "cycle_result": sample["cycle_result"],
    }
    row.update({field: sample[field] for field in PHASES + ("total_cycles",)})
    row.update({
        "core_run_us": sample["compute_cycles"] / frequency,
        "total_us": total / frequency,
        "effective_macs_per_second": macs * frequency * 1_000_000.0 / total,
        "effective_mmacs_per_second": macs * frequency / total,
        "core_effective_mmacs_per_second": macs * frequency / sample["compute_cycles"],
        "peak_issue_mmacs_per_second": frequency,
        "active_mac_utilization": macs / sample["compute_cycles"],
        "overall_mac_utilization": macs / total,
        "io_fraction": io_cycles / total,
        "host_wait_fraction": sample["host_wait_cycles"] / total,
        "setup_fraction": sample["setup_cycles"] / total,
        "core_fraction": sample["compute_cycles"] / total,
        "amortized_weight_load_cycles_per_transaction": sample["weight_load_cycles"] / sample["reuse"],
        "cycles_per_transaction": total / sample["transactions"],
        "sample_key": "{}_r{}_h{}".format(sample["kernel"], sample["reuse"], sample["host_wait"]),
        "model_results_file": profile["analysis_template"] + "/model_results.json",
    })
    return row


def assert_template_equivalence(rows: List[Dict[str, Any]]) -> None:
    """A template's area metadata must never create an ESL performance advantage."""
    groups = {}
    fields = PHASES + ("total_cycles", "macs", "core_run_us", "total_us",
                       "effective_macs_per_second", "active_mac_utilization", "overall_mac_utilization")
    for row in rows:
        key = (row["kernel"], row["reuse"], row["host_wait"], row["assumed_frequency_mhz"])
        values = tuple(row[field] for field in fields)
        if key in groups and groups[key] != values:
            raise AnalysisError("Template-dependent performance appeared in common core model: {}".format(key))
        groups[key] = values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def write_chart(path: Path, rows: List[Dict[str, Any]]) -> None:
    chosen = [row for row in rows if row["template"] == TEMPLATES[0]
              and row["assumed_frequency_mhz"] == 25
              and (row["reuse"], row["host_wait"]) in ((1, 0), (16, 0), (16, 16))]
    chosen.sort(key=lambda row: (KERNELS.index(row["kernel"]), row["reuse"], row["host_wait"]))
    colors = ("#7c3aed", "#0284c7", "#64748b", "#16a34a", "#d97706", "#dc2626")
    labels = ("weight load", "input load", "START/setup", "core RUN", "readout", "host wait")
    width, left, plot_width, top, row_height = 1180, 265, 690, 145, 76
    height = top + len(chosen) * row_height + 110
    maximum = max(row["cycles_per_transaction"] for row in chosen)
    axis_max = math.ceil(maximum / 5) * 5
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
        width, height, width, height), '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172033">',
        '<text x="32" y="35" font-size="23" font-weight="bold">Tiny-LPU ESL: cycles per transaction</text>',
        '<text x="32" y="64" font-size="14">Common core model for SKY / IHP / GF; frequency is assumed, not measured or STA-validated.</text>']
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = 32 + index * 176
        svg.append('<rect x="{}" y="88" width="13" height="13" fill="{}"/>'.format(x, color))
        svg.append('<text x="{}" y="99" font-size="13">{}</text>'.format(x + 19, label))
    for tick in range(0, int(axis_max) + 1, 5):
        x = left + plot_width * tick / axis_max
        svg.append('<line x1="{0:.2f}" y1="{1}" x2="{0:.2f}" y2="{2}" stroke="#e2e8f0"/>'.format(
            x, top - 9, top + len(chosen) * row_height - 18))
        svg.append('<text x="{:.2f}" y="{}" text-anchor="middle" font-size="12">{}</text>'.format(
            x, top + len(chosen) * row_height + 5, tick))
    for index, row in enumerate(chosen):
        y = top + index * row_height
        label = "{} | reuse={} | wait={}".format(row["kernel"], row["reuse"], row["host_wait"])
        svg.append('<text x="32" y="{}" font-size="14">{}</text>'.format(y + 20, html.escape(label)))
        x = float(left)
        for phase, color in zip(PHASES, colors):
            cycles = row[phase] / row["transactions"]
            segment_width = plot_width * cycles / axis_max
            svg.append('<rect x="{:.3f}" y="{}" width="{:.3f}" height="30" fill="{}"><title>{}: {:.3f} cycles/transaction</title></rect>'.format(
                x, y, segment_width, color, phase, cycles))
            x += segment_width
        svg.append('<text x="{:.2f}" y="{}" font-size="13">{:.3f} cycles</text>'.format(x + 9, y + 20, row["cycles_per_transaction"]))
        svg.append('<text x="{}" y="{}" font-size="12" fill="#475569">{:.3f} MMAC/s at assumed 25 MHz; total run: {} cycles / {} transactions</text>'.format(
            left, y + 48, row["effective_mmacs_per_second"], row["total_cycles"], row["transactions"]))
    svg.append('<text x="{}" y="{}" text-anchor="middle" font-size="14">clock cycles / transaction (one-time weight load amortized)</text>'.format(
        left + plot_width / 2, height - 32))
    svg.append('</g></svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def make_report(rows: List[Dict[str, Any]], profiles: List[Dict[str, Any]], template: str = None) -> str:
    prefix = "../" if template else ""
    title = "Tiny-LPU ESL performance analysis" + (" — " + template.upper() if template else " - three-template summary")
    lines = ["# " + title, "", "This report uses functional/cycle models. Frequencies are ESL assumptions; this runner performs no RTL synthesis, routing or STA.",
             "SKY, IHP and GF each load their own pinned template metadata and instantiate the same core model. Equal workloads/resources/clocks give equal cycles; PDK, tile and nominal clock are metadata.", "",
             "## Model and accounting", "",
             "- A=[1,-2,3,-4]; dot4 W=[7,-8,-1,2] gives [12]; the second GEMV row [-8,7,0,-1] gives [12,-18].",
             "- M is output rows (dot4=1, GEMV=2); N is transactions/weight reuse; H is extra host-wait clocks per transaction. Load W once, A each transaction.",
             "- An 8-bit load transfers two INT4 values per clock; each INT16 output takes two readout clocks. A row executes a five-clock dot4; GEMV rows are serial.",
             "- C_total = 2M + N*(3 + H + 7M): weights=2M, inputs=2N, setup=N, RUN=5MN, readout=2MN, wait=HN. Phases do not overlap.",
             "- t_us=cycles/f_MHz; MMAC/s=MACs*f_MHz/C_total. One MAC counts one multiply-accumulate.",
             "- Active utilization=MACs/RUN=0.8; overall utilization=MACs/C_total. Peak issue is one MAC/clock, not one dot4/clock.", "",
             "## Template metadata", "",
             "| Template | PDK | Action | Nominal period ns | Tile DIE_AREA um2 | Physical performance validated by this runner |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for profile in profiles:
        lines.append("| {} | {} | {} | {} | {:.2f} | false |".format(
            profile["analysis_template"], profile["pdk"], profile["action_ref"], profile["clock_period_ns"], profile["tile_die_area_um2"]))
    lines.extend(["", "Tile boundary area does not prove usable cell area, fit or Fmax. This study does not invent technology-dependent cycle differences.", "",
                  "## Representative 25MHz scenarios", "",
                  "| Kernel | Reuse | Wait/transaction | RUN clocks | Total clocks | RUN us | Total us | MMAC/s | I/O fraction | Host-wait fraction | Overall utilization | Weight clocks/transaction |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    representative = template or TEMPLATES[0]
    for row in rows:
        if row["template"] != representative or row["assumed_frequency_mhz"] != 25:
            continue
        if (row["reuse"], row["host_wait"]) not in ((1, 0), (16, 0), (16, 16)):
            continue
        lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1%} | {:.1%} | {:.1%} | {:.3f} |".format(
            row["kernel"], row["reuse"], row["host_wait"], row["compute_cycles"], row["total_cycles"],
            row["core_run_us"], row["total_us"], row["effective_mmacs_per_second"], row["io_fraction"],
            row["host_wait_fraction"], row["overall_mac_utilization"], row["amortized_weight_load_cycles_per_transaction"]))
    lines.extend(["", "![Cycles per transaction](" + prefix + "workload_breakdown.svg)", "",
                  "## Conclusions", "",
                  "1. With no extra wait and one transaction, dot4 takes 12 clocks and GEMV 21; RUN is only 5/10 clocks. End-to-end throughput is below core throughput.",
                  "2. Reuse=16 gives dot4/GEMV totals of 162/276 clocks, or 10.125/17.25 per transaction. Weight load amortizes, but input/start/readout remain per transaction.",
                  "3. Adding 16 host-wait clocks per transaction raises those totals to 418/532. Weight reuse does not remove host wait.",
                  "4. At fixed cycles, 10/25/50MHz scales time and throughput only. This does not establish technology Fmax, power, physical area or I/O setup/hold.", "",
                  "## Reproducible data", ""])
    if template:
        lines.extend(["- [Template performance table](performance.csv)", "- [Template metadata, functional output and raw model JSON](model_results.json)",
                      "- Baseline traces: [dot4](traces/dot4.csv), [GEMV](traces/gemv2x4.csv)"])
    else:
        lines.extend(["- [Full CSV](summary.csv), [structured JSON](summary.json)",
                      "- Template reports: " + ", ".join("[{}]({}/analysis.md)".format(name.upper(), name) for name in TEMPLATES)])
    lines.extend(["", "Default sweep: 3 templates x 2 kernels x 3 reuse values (1/4/16) x 3 host waits (0/4/16) x 3 MHz assumptions (10/25/50) = 162 rows.",
                  "The quick sweep uses reuse=1/16, wait=0/16, 25MHz and 24 rows. Nonzero model exits, invalid JSON, incorrect outputs or cycle counts fail the runner.", ""])
    return "\n".join(lines)


def analyze(model: Path, root: Path, output: Path, quick: bool = False) -> List[Dict[str, Any]]:
    profiles = load_profiles(root)
    if not model.is_file():
        raise AnalysisError("Model executable not found: {}".format(model))
    output.mkdir(parents=True, exist_ok=True)
    reuses = (1, 16) if quick else (1, 4, 16)
    waits = (0, 16) if quick else (0, 4, 16)
    frequencies = (25,) if quick else (10, 25, 50)
    all_rows = []
    for profile in profiles:
        template = profile["analysis_template"]
        samples = []
        rows = []
        for kernel in KERNELS:
            for reuse in reuses:
                for wait in waits:
                    trace = output / template / "traces" / (kernel + ".csv") if reuse == 1 and wait == 0 else None
                    sample = run_model(model, kernel, reuse, wait, trace)
                    samples.append({"sample_key": "{}_r{}_h{}".format(kernel, reuse, wait),
                                    "inputs": {"a": list(INPUT_A), "w": list(INPUT_W[kernel])},
                                    "cli_output": sample})
                    rows.extend(make_metrics(profile, sample, frequency) for frequency in frequencies)
        write_json(output / template / "model_results.json", {
            "profile": profile, "model_executable": str(model), "samples": samples,
            "physical_performance_validated": False,
        })
        write_csv(output / template / "performance.csv", rows)
        (output / template / "analysis.md").write_text(make_report(rows, [profile], template), encoding="utf-8")
        all_rows.extend(rows)
    assert_template_equivalence(all_rows)
    write_csv(output / "summary.csv", all_rows)
    write_json(output / "summary.json", {
        "schema_version": 1, "analysis_kind": "ESL assumptions, not physical measurements",
        "physical_performance_validated": False, "model_executable": str(model),
        "quick": quick, "profiles": profiles, "row_count": len(all_rows), "rows": all_rows,
    })
    write_chart(output / "workload_breakdown.svg", all_rows)
    (output / "performance_analysis.md").write_text(make_report(all_rows, profiles), encoding="utf-8")
    return all_rows


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="compiled tinylpu_esl executable")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    parser.add_argument("--output", type=Path, help="output directory; default ROOT/reports/esl")
    parser.add_argument("--quick", action="store_true", help="24 rows instead of the default 162")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "reports" / "esl"
    try:
        rows = analyze(args.model.resolve(), root, output, args.quick)
    except AnalysisError as exc:
        print("ESL analysis failed: {}".format(exc), file=sys.stderr)
        return 1
    print("ESL analysis passed: {} rows; reports in {}".format(len(rows), output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
