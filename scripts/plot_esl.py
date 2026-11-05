#!/usr/bin/env python3
"""Optional Matplotlib PNG for the common-core ESL performance report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "reports/esl/summary.json")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("physical_performance_validated") is not False:
        raise ValueError("This chart requires unvalidated ESL assumptions to be labeled explicitly")
    selected = [r for r in report["rows"] if r["template"] == "sky"
                and r["assumed_frequency_mhz"] == 25
                and (r["reuse"], r["host_wait"]) in ((1, 0), (16, 0), (16, 16))]
    selected.sort(key=lambda r: (r["kernel"] != "dot4", r["reuse"], r["host_wait"]))
    if len(selected) != 6:
        raise ValueError("Expected the six representative dot4/GEMV scenarios")
    phases = ["weight_load_cycles", "input_load_cycles", "setup_cycles",
              "compute_cycles", "readout_cycles", "host_wait_cycles"]
    labels = ["Weight load", "Input load", "START", "Core RUN", "Readout", "Host wait"]
    colors = ["#7c3aed", "#0284c7", "#64748b", "#16a34a", "#d97706", "#dc2626"]
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=160)
    left = [0.0] * len(selected)
    for phase, label, color in zip(phases, labels, colors):
        widths = [r[phase] / r["transactions"] for r in selected]
        ax.barh(range(len(selected)), widths, left=left, height=0.53,
                label=label, color=color)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(["{} | reuse={} | wait={}\n{:.2f} MMAC/s at assumed 25 MHz".format(
        r["kernel"], r["reuse"], r["host_wait"], r["effective_mmacs_per_second"])
        for r in selected], fontsize=10)
    for index, value in enumerate(left):
        ax.text(value + 0.45, index, "{:g}".format(value), va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 39)
    ax.set_xlabel("Clock cycles per transaction (one-time weight load amortized)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10)
    fig.suptitle("Tiny-LPU ESL: data movement and host waiting dominate", x=0.025,
                 y=0.975, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.025, 0.925, "Same core model for SKY / IHP / GF. Frequency is an assumption; no STA or physical fit result.", fontsize=10)
    fig.text(0.025, 0.02, "RUN utilization = 80%. Total time includes load, START, RUN, readout and host wait. 1 MAC = one multiply-accumulate.", fontsize=9)
    fig.subplots_adjust(left=0.31, right=0.975, top=0.79, bottom=0.12)
    target = args.report.parent / "workload_breakdown.png"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
