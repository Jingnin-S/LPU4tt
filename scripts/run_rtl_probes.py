#!/usr/bin/env python3
"""Run the small RTL probes with available server tools.

The script has three deliberately separate evidence classes:

* VCS/iverilog/Verilator simulation checks the probe behavior against the
  self-checking testbench.
* Yosys, when available, performs generic ``memory_map``/``techmap`` and
  writes JSON cell statistics.  These are technology-independent resources.
* Synopsys Design Compiler, when available through Modules, maps a small
  representative set to the checked-in FreePDK45/gscl45nm Liberty. Reports
  include the exact library and 10 ns constraint, but this script never turns
  cell counts or slack into an unqualified Fmax claim.

Missing tools are recorded explicitly. A simulator failure is fatal; missing
Yosys/STA does not silently become a physical signoff result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl" / "tinylpu_memory_mac_probe.sv"
TB = ROOT / "rtl" / "tb_tinylpu_memory_mac_probe.sv"
LIBERTY = Path(os.environ.get(
    "TINYLPU_PROBE_LIBERTY",
    "/tools/PDK/NCSU45PDK/FreePDK45/osu_soc/lib/files/gscl45nm.lib"))
DB = Path(os.environ.get(
    "TINYLPU_PROBE_DB",
    "/tools/PDK/NCSU45PDK/FreePDK45/osu_soc/lib/files/gscl45nm.db"))


def run(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_path(tool: str) -> str | None:
    env_name = {
        "yosys": "TINYLPU_YOSYS",
        "dc_shell": "TINYLPU_DC_SHELL",
    }.get(tool)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    if tool == "yosys":
        return shutil.which("yosys") or shutil.which("yowasp-yosys")
    return shutil.which(tool)


def tool_version(tool: str) -> str | None:
    path = tool_path(tool)
    if path is None:
        return None
    if tool == "vcs":
        probe = [path, "-ID"]
    elif tool == "yosys":
        probe = [path, "-V"]
    else:
        probe = [path, "-version"]
    result = run(probe)
    text = (result["stdout"] + result["stderr"]).strip().splitlines()
    return text[0] if text else path


def compile_simulator(outdir: Path) -> dict[str, Any]:
    if tool_path("iverilog"):
        return {
            "kind": "iverilog",
            "executable": outdir / "probe.vvp",
            "compile": run(["iverilog", "-g2012", "-s", "tb_tinylpu_memory_mac_probe",
                             "-o", str(outdir / "probe.vvp"), str(RTL), str(TB)]),
        }
    if tool_path("verilator"):
        return {
            "kind": "verilator",
            "executable": outdir / "probe_sim",
            "compile": run(["verilator", "--binary", "--timing", "--top-module",
                             "tb_tinylpu_memory_mac_probe", str(RTL), str(TB),
                             "-o", str(outdir / "probe_sim")]),
        }
    if tool_path("vcs"):
        return {
            "kind": "vcs",
            "executable": outdir / "probe_sim",
            "compile": run(["vcs", "-full64", "-sverilog", "-timescale=1ns/1ps",
                             "-top", "tb_tinylpu_memory_mac_probe", str(RTL), str(TB),
                             "-o", str(outdir / "probe_sim")]),
        }
    return {
        "kind": None,
        "executable": None,
        "compile": {"command": [], "returncode": None, "stdout": "",
                     "stderr": "no iverilog/verilator/vcs found"},
    }


def generic_settings(kind: str, parameter: int,
                     extra: int | None) -> tuple[str, list[tuple[str, int]]]:
    if kind == "flat":
        return "tinylpu_flat_byte_array_probe", [("CAPACITY", parameter),
                                                  ("READ_PORTS", extra or 1)]
    if kind == "bank":
        return "tinylpu_banked_byte_array_probe", [("CAPACITY", parameter),
                                                   ("BANKS", extra or 2)]
    return "tinylpu_int4_mac_probe", [("LANES", parameter),
                                       ("ACC_BITS", extra or 16)]


def synthesize_generic(kind: str, parameter: int, extra: int | None,
                       outdir: Path) -> dict[str, Any]:
    yosys = tool_path("yosys")
    label = f"{kind}_{parameter}_{extra or 0}"
    base: dict[str, Any] = {"kind": kind, "parameter": parameter,
                            "extra": extra, "label": label}
    if yosys is None:
        base.update({"available": False, "reason": "yosys not found",
                     "cell_stats": None})
        return base
    top, settings = generic_settings(kind, parameter, extra)
    stats = outdir / f"{label}.json"
    ys = outdir / f"{label}.ys"
    lines = [f'read_verilog -sv "{RTL}"']
    for name, value in settings:
        lines.append(f"chparam -set {name} {value} {top}")
    lines += [
        f"hierarchy -top {top}",
        "proc",
        "memory_map",
        "opt",
        "techmap",
        "opt",
        "abc -g simple",
        f'tee -o "{stats}" stat -json',
    ]
    ys.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = run([yosys, "-s", str(ys)])
    base.update({"available": True, "result": result,
                 "metric_note": "generic Yosys memory_map/techmap/ABC cells; no Liberty/STA"})
    if stats.exists():
        try:
            base["cell_stats"] = json.loads(stats.read_text(encoding="utf-8"))
            counts = base["cell_stats"].get("design", {}).get("num_cells_by_type", {})
            dff = sum(value for name, value in counts.items() if "DFF" in name)
            mux = sum(value for name, value in counts.items() if "MUX" in name)
            logic = sum(value for name, value in counts.items()
                        if "DFF" not in name and "MUX" not in name)
            base["cell_summary"] = {
                "cell_total": base["cell_stats"].get("design", {}).get("num_cells"),
                "dff": dff,
                "mux": mux,
                "logic": logic,
            }
        except json.JSONDecodeError as exc:
            base["cell_stats_error"] = str(exc)
    else:
        base["cell_stats"] = None
    return base


def addr_bits(capacity: int) -> int:
    return max(1, math.ceil(math.log2(capacity)))


def make_wrapper(kind: str, parameter: int, extra: int, path: Path) -> str:
    if kind == "flat":
        ports = extra
        bits = addr_bits(parameter)
        text = f"""module probe_top(
  input wire clk, input wire rst_n, input wire wr_en,
  input wire [7:0] wr_addr, input wire [7:0] wr_data,
  input wire [{ports * 8 - 1}:0] rd_addr,
  output wire [{ports * 8 - 1}:0] rd_data
);
  tinylpu_flat_byte_array_probe #(.CAPACITY({parameter}), .READ_PORTS({ports}),
      .ADDR_BITS({bits})) u_probe(.clk(clk), .rst_n(rst_n), .wr_en(wr_en),
      .wr_addr(wr_addr), .wr_data(wr_data), .rd_addr(rd_addr), .rd_data(rd_data));
endmodule
"""
    elif kind == "bank":
        banks = extra
        bits = addr_bits(parameter)
        text = f"""module probe_top(
  input wire clk, input wire rst_n, input wire wr_en,
  input wire [7:0] wr_addr, input wire [7:0] wr_data,
  input wire [{banks * 8 - 1}:0] rd_addr,
  output wire [{banks * 8 - 1}:0] rd_data
);
  tinylpu_banked_byte_array_probe #(.CAPACITY({parameter}), .BANKS({banks}),
      .ADDR_BITS({bits})) u_probe(.clk(clk), .rst_n(rst_n), .wr_en(wr_en),
      .wr_addr(wr_addr), .wr_data(wr_data), .rd_addr(rd_addr), .rd_data(rd_data));
endmodule
"""
    else:
        lanes, acc_bits = parameter, extra
        text = f"""module probe_top(
  input wire clk, input wire rst_n, input wire clear_acc,
  input wire issue_valid, input wire issue_last,
  input wire [{lanes * 4 - 1}:0] a_in,
  input wire [{lanes * 4 - 1}:0] w_in,
  output wire done,
  output wire [{lanes * acc_bits - 1}:0] acc_out
);
  tinylpu_int4_mac_probe #(.LANES({lanes}), .ACC_BITS({acc_bits})) u_probe(
      .clk(clk), .rst_n(rst_n), .clear_acc(clear_acc),
      .issue_valid(issue_valid), .issue_last(issue_last), .a_in(a_in),
      .w_in(w_in), .done(done), .acc_out(acc_out));
endmodule
"""
    path.write_text(text, encoding="utf-8")
    return text


def parse_dc_area(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total cell area:"):
            values["total_cell_area"] = stripped.split()[-1]
        elif stripped.startswith("Combinational area:"):
            values["combinational_area"] = stripped.split()[-1]
        elif stripped.startswith("Noncombinational area:"):
            values["noncombinational_area"] = stripped.split()[-1]
        elif stripped.startswith("Number of cells:"):
            values["number_of_cells"] = stripped.split()[-1]
    return values


def synthesize_dc(kind: str, parameter: int, extra: int, outdir: Path,
                  tempdir: Path) -> dict[str, Any]:
    dc_shell = tool_path("dc_shell")
    label = f"{kind}_{parameter}_{extra}"
    result: dict[str, Any] = {"kind": kind, "parameter": parameter,
                              "extra": extra, "label": label,
                              "library": str(LIBERTY), "db": str(DB),
                              "library_corner": "gscl45nm typical, 1.1 V, 27 C",
                              "clock_period_ns": 10.0,
                              "wire_assumption": "unrouted/no parasitics"}
    if dc_shell is None:
        result.update({"available": False, "reason": "dc_shell not found"})
        return result
    wrapper = tempdir / f"{label}_wrapper.sv"
    make_wrapper(kind, parameter, extra, wrapper)
    area = outdir / f"{label}.area.rpt"
    timing = outdir / f"{label}.timing.rpt"
    mapped = outdir / f"{label}.mapped.v"
    tcl = tempdir / f"{label}.tcl"
    tcl.write_text(f"""set_app_var search_path [list {ROOT / 'rtl'} {tempdir}]
set target_library [list {DB}]
set link_library [list * {DB}]
set_host_options -max_cores 2
analyze -format sverilog [list {RTL} {wrapper}]
elaborate probe_top
current_design probe_top
link
uniquify
create_clock -name clk -period 10 [get_ports clk]
set non_clock_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.5 -clock clk $non_clock_inputs
set_output_delay 0.5 -clock clk [all_outputs]
set_dont_touch_network [get_ports rst_n]
compile -map_effort medium
report_area -hierarchy > {area}
report_timing -max_paths 5 > {timing}
write -format verilog -hierarchy -output {mapped}
quit
""", encoding="utf-8")
    command = [dc_shell, "-64bit", "-f", str(tcl)]
    run_result = run(command)
    log = outdir / f"{label}.dc.log"
    log.write_text(run_result["stdout"] + "\n" + run_result["stderr"], encoding="utf-8")
    result.update({"available": True, "result": run_result,
                   "area_report": str(area), "timing_report": str(timing),
                   "mapped_netlist": str(mapped),
                   "area": parse_dc_area(area.read_text(encoding="utf-8") if area.exists() else ""),
                   "metric_note": "mapped gscl45nm Liberty area; constrained 10 ns report, not an Fmax claim"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "architecture" / "rtl")
    parser.add_argument("--yosys", help="explicit yosys/yowasp-yosys executable")
    parser.add_argument("--dc-shell", help="explicit dc_shell executable")
    parser.add_argument("--liberty", help="mapped Liberty path (or TINYLPU_PROBE_LIBERTY)")
    parser.add_argument("--db", help="mapped .db path (or TINYLPU_PROBE_DB)")
    parser.add_argument("--generic-only", action="store_true",
                        help="run/preserve generic sweep without mapped DC")
    parser.add_argument("--mapped-only", action="store_true",
                        help="run/preserve mapped DC without generic sweep")
    args = parser.parse_args()
    if args.generic_only and args.mapped_only:
        parser.error("--generic-only and --mapped-only are mutually exclusive")
    if args.yosys:
        os.environ["TINYLPU_YOSYS"] = args.yosys
    if args.dc_shell:
        os.environ["TINYLPU_DC_SHELL"] = args.dc_shell
    global LIBERTY, DB
    if args.liberty:
        LIBERTY = Path(args.liberty)
    if args.db:
        DB = Path(args.db)
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "rtl_sources": [str(RTL.relative_to(ROOT)), str(TB.relative_to(ROOT))],
        "rtl_source_sha256": {
            str(RTL.relative_to(ROOT)): sha256_file(RTL),
            str(TB.relative_to(ROOT)): sha256_file(TB),
        },
        "tools": {tool: tool_version(tool) for tool in
                  ("yosys", "iverilog", "verilator", "vcs", "dc_shell", "opensta", "sta")},
        "simulation": None,
        "generic_synthesis": [],
        "mapped_synthesis": [],
        "physical_performance_validated": False,
        "limitations": [
            "No Fmax is inferred from generic RTL, cell counts, or a single constrained report.",
            "Mapped reports use the exact gscl45nm Liberty path and 10 ns clock constraint; "
            "they are probe evidence, not Tiny-LPU chip signoff.",
            "The banked probe exposes one registered read per bank and does not model a "
            "cross-bank gather/arbiter or the complete transport top.",
            "Routing, parasitics, IR drop, DRC/LVS, and final pad integration are not modeled.",
        ],
    }
    previous = args.output / "summary.json"
    if previous.exists() and (args.generic_only or args.mapped_only):
        try:
            old = json.loads(previous.read_text(encoding="utf-8"))
            if args.generic_only:
                result["mapped_synthesis"] = old.get("mapped_synthesis", [])
            if args.mapped_only:
                result["generic_synthesis"] = old.get("generic_synthesis", [])
        except (OSError, json.JSONDecodeError):
            pass
    with tempfile.TemporaryDirectory(prefix="tinylpu_rtl_probe_") as temp:
        tempdir = Path(temp)
        sim = compile_simulator(tempdir)
        compile_result = sim["compile"]
        if compile_result.get("returncode") == 0:
            sim["run"] = run([str(sim["executable"])])
        result["simulation"] = sim

        if not args.mapped_only:
            # Generic full sweep: all requested capacities and read-port/bank
            # variants, plus both accumulator widths and all lane counts.
            for capacity in (32, 64, 96, 128, 192, 256):
                for ports in (1, 2, 3):
                    result["generic_synthesis"].append(
                        synthesize_generic("flat", capacity, ports, args.output))
                for banks in (2, 4):
                    result["generic_synthesis"].append(
                        synthesize_generic("bank", capacity, banks, args.output))
            for lanes in (1, 2, 4):
                for acc_bits in (16, 32):
                    result["generic_synthesis"].append(
                        synthesize_generic("mac", lanes, acc_bits, args.output))

        if not args.generic_only:
            # Mapped representatives emphasize the 64/128/192 capacity decisions
            # and both MAC accumulator widths without a large commercial matrix.
            for capacity in (64, 128, 192):
                for ports in (1, 2, 3):
                    result["mapped_synthesis"].append(
                        synthesize_dc("flat", capacity, ports, args.output, tempdir))
                for banks in (2, 4):
                    result["mapped_synthesis"].append(
                        synthesize_dc("bank", capacity, banks, args.output, tempdir))
            for lanes in (1, 2, 4):
                for acc_bits in (16, 32):
                    result["mapped_synthesis"].append(
                        synthesize_dc("mac", lanes, acc_bits, args.output, tempdir))

    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    sim_result = result["simulation"]
    sim_rc = None if sim_result is None else sim_result["compile"].get("returncode")
    if sim_result is not None and sim_result.get("run") is not None:
        sim_rc = sim_result["run"].get("returncode")
    mapped_ok = sum(1 for item in result["mapped_synthesis"]
                    if item.get("available") and item.get("result", {}).get("returncode") == 0)
    print(json.dumps({
        "simulation_returncode": sim_rc,
        "yosys": result["tools"]["yosys"],
        "dc_shell": result["tools"]["dc_shell"],
        "generic_synthesis_runs": len(result["generic_synthesis"]),
        "mapped_synthesis_runs": len(result["mapped_synthesis"]),
        "mapped_synthesis_passed": mapped_ok,
        "physical_performance_validated": False,
    }, sort_keys=True))
    # A missing simulator or failed simulation is a real probe failure. A
    # missing generic Yosys/STA is captured as a limitation, not a fake pass.
    return 0 if sim_rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
