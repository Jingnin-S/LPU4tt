#!/usr/bin/env python3
"""Optional Matplotlib plots of the ESL design-space CSV."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "reports/dse/summary.csv")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(args.report.open(encoding="utf-8")))
    rows = [r for r in rows if r["template"] == "sky" and r["order"] == "weight_stationary"]

    def find(workload, architecture, capacity, frequency=100, bandwidth=25):
        return next((float(r["effective_mmacs_per_second"]) for r in rows
                     if r["workload"] == workload and r["architecture"] == architecture
                     and int(r["scratchpad_capacity_bytes"]) == capacity
                     and int(r["core_mhz"]) == frequency and int(r["io_MBps"]) == bandwidth), float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=180)
    colors = ["#0284c7", "#7c3aed", "#16a34a"]
    freq = [50, 100, 150, 200]
    for bandwidth, color in zip((1, 5, 25), colors):
        axes[0].plot(freq, [find("fc4x16_b16", "stream2_ping", 64, f, bandwidth) for f in freq],
                     marker="o", color=color, label="{} MB/s host".format(bandwidth))
    axes[0].set(title="Clock gains meet the IO ceiling", xlabel="Assumed core frequency (MHz)", ylabel="End-to-end MMAC/s")
    axes[0].set_xticks(freq)
    axes[0].legend(fontsize=9)
    axes[0].text(0.04, 0.62, "FC4x16, batch 16\n2 lanes, 64 B, double buffer", transform=axes[0].transAxes, fontsize=9)
    caps = [16, 32, 64, 128, 256]
    for lanes, color in zip((1, 2, 4), colors):
        axes[1].plot(caps, [find("fc8x32_b16", "stream{}_ping".format(lanes), c) for c in caps],
                     marker="o", color=color, label="{} lane(s)".format(lanes))
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(caps)
    axes[1].set_xticklabels(caps)
    axes[1].set(title="Storage removes tiling traffic", xlabel="Scratchpad capacity budget (B)", ylabel="End-to-end MMAC/s")
    axes[1].text(0.04, 0.62, "FC8x32, batch 16\n100 MHz, 25 MB/s\nMissing points: does not fit", transform=axes[1].transAxes, fontsize=9)
    axes[1].legend(fontsize=9)
    workloads = [("fir8_b16", 32), ("fc4x16_b16", 64), ("fc8x32_b16", 256)]
    for offset, (lanes, color) in enumerate(zip((1, 2, 4), colors)):
        axes[2].bar([i + (offset - 1) * 0.25 for i in range(3)],
                    [find(w, "stream{}_ping".format(lanes), c) for w, c in workloads],
                    width=0.24, color=color, label="{} lane(s)".format(lanes))
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(["FIR8\n32 B", "FC4x16\n64 B", "FC8x32\n256 B"])
    axes[2].set(title="PE count must match the workload", ylabel="End-to-end MMAC/s")
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8f0")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Tiny-LPU: frequency, storage and workload trade-offs", x=0.025, y=0.98, ha="left", fontweight="bold", fontsize=15)
    fig.text(0.025, 0.91, "Common SKY / IHP / GF ESL model. 100 MHz is a target; no STA, power or tile-fit result.", fontsize=10)
    fig.text(0.025, 0.025, "All panels include host transfer and result readout. 3-stage pipelines; batch=16; double buffering. Data capacity excludes pipeline/control state.", fontsize=9)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.065, right=0.985, wspace=0.32)
    target = args.report.parent / "tradeoffs.png"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
