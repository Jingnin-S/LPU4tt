#!/usr/bin/env python3
"""Small mapped timing-constraint sweep for representative RTL probes.

This is a constraint sensitivity report only.  It uses the checked-in probe
RTL and the caller-selected Liberty/.db; it does not claim routed timing or a
chip Fmax.  The full generic/mapped area sweep remains in run_rtl_probes.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

import run_rtl_probes as probe


def timing_value(text: str) -> str | None:
    values = []
    for line in text.splitlines():
        if "slack (" in line:
            match = re.search(r"slack \([^)]*\)\s+([-+]?\d+(?:\.\d+)?)", line)
            if match:
                values.append(match.group(1))
    return min(values, key=lambda value: float(value)) if values else None


def run_case(kind: str, parameter: int, extra: int, period: float,
             output: Path, tempdir: Path) -> dict:
    dc = probe.tool_path("dc_shell")
    label = f"{kind}_{parameter}_{extra}_{str(period).replace('.', 'p')}ns"
    result = {
        "label": label,
        "kind": kind,
        "parameter": parameter,
        "extra": extra,
        "period_ns": period,
        "library": str(probe.LIBERTY),
        "db": str(probe.DB),
        "corner": "gscl45nm typical, 1.1 V, 27 C",
        "wire_assumption": "unrouted/no parasitics",
    }
    if dc is None:
        result.update({"returncode": None, "reason": "dc_shell not found"})
        return result
    wrapper = tempdir / f"{label}_wrapper.sv"
    probe.make_wrapper(kind, parameter, extra, wrapper)
    timing = output / f"{label}.timing.rpt"
    tcl = tempdir / f"{label}.tcl"
    tcl.write_text(f"""set_app_var search_path [list {probe.ROOT / 'rtl'} {tempdir}]
set target_library [list {probe.DB}]
set link_library [list * {probe.DB}]
set_host_options -max_cores 2
analyze -format sverilog [list {probe.RTL} {wrapper}]
elaborate probe_top
current_design probe_top
link
uniquify
create_clock -name clk -period {period} [get_ports clk]
set non_clock_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.5 -clock clk $non_clock_inputs
set_output_delay 0.5 -clock clk [all_outputs]
set_dont_touch_network [get_ports rst_n]
compile -map_effort medium
report_timing -max_paths 5 > {timing}
quit
""", encoding="utf-8")
    command_result = probe.run([dc, "-64bit", "-f", str(tcl)])
    text = command_result["stdout"] + "\n" + command_result["stderr"]
    report_text = timing.read_text(encoding="utf-8") if timing.exists() else ""
    result.update({"returncode": command_result["returncode"],
                   "worst_reported_slack_ns": timing_value(report_text),
                   "timing_report": str(timing),
                   "stderr_tail": command_result["stderr"][-1000:]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=probe.ROOT / "reports" / "architecture" / "rtl" / "timing_sweep")
    parser.add_argument("--liberty")
    parser.add_argument("--db")
    args = parser.parse_args()
    # Keep the helper's defaults for the normal server path; explicit
    # overrides update the imported module before cases are generated.
    if args.liberty:
        probe.LIBERTY = Path(args.liberty)
    if args.db:
        probe.DB = Path(args.db)
    args.output.mkdir(parents=True, exist_ok=True)
    configs = [("flat", 128, 3), ("bank", 128, 4),
               ("bank", 192, 4), ("mac", 2, 16), ("mac", 2, 32)]
    results = []
    with tempfile.TemporaryDirectory(prefix="tinylpu_rtl_timing_") as temp:
        tempdir = Path(temp)
        for period in (10.0, 6.667, 5.0):
            for kind, parameter, extra in configs:
                results.append(run_case(kind, parameter, extra, period,
                                        args.output, tempdir))
    summary = {
        "configs": configs,
        "periods_ns": [10.0, 6.667, 5.0],
        "results": results,
        "physical_performance_validated": False,
        "note": "Constraint sensitivity only; gscl45nm typical, unrouted/no parasitics; no Fmax claim.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
    failed = [item for item in results if item.get("returncode") != 0]
    print(json.dumps({"total": len(results), "passed": len(results)-len(failed),
                      "failed": len(failed), "physical_performance_validated": False},
                     sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
