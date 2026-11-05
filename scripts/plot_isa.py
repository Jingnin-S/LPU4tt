#!/usr/bin/env python3
"""Derived standalone figure; source CSVs stay unchanged."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"reports/isa"


def main():
    with (FOLDER/"comparison.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (FOLDER/"ablations/comparison.csv").open(encoding="utf-8") as stream:
        ablations = {row["case"]: row for row in csv.DictReader(stream)}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1, .85]})
    for k, marker in ((16, "o"), (64, "s"), (256, "^")):
        selected = [r for r in rows if r["fixture"] == f"linear_{k}x8" and r["frames"] == "8"
                    and r["input_bits"] == "4" and r["output_bits"] == "32"
                    and r["order"] == "frame_major" and r["encoding"] == "4"]
        selected.sort(key=lambda r: int(r["scratch_bytes"]))
        axes[0].plot([int(r["scratch_bytes"]) for r in selected],
                     [float(r["time_us"])/8 for r in selected], marker=marker, label=f"M=8, K={k}")
    axes[0].set(xlabel="Data scratchpad (bytes)", ylabel="Cold batch time / frame (us)",
                title="Memory and tiling trade-off", yscale="log")
    axes[0].set_xticks([32, 64, 96, 128, 192, 256])
    axes[0].grid(axis="both", alpha=.25)
    axes[0].set_ylim(6, 500)
    axes[0].legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center")
    labels = ["Fused MVM\nW resident", "Row groups\nunrolled", "W reload\neach frame",
              "Device\nK accumulation", "Host sums\nK partials"]
    keys = ["fused_rows_resident", "unrolled_row_groups_same_data", "reload_weights",
            "device_accumulate", "host_partial_sum_same_chunks"]
    values = [ablations[key] for key in keys]
    components = (("device_cycles", .01, "Device cycles", "#2f6f9f"),
                                        ("command_bytes", .1, "Command link", "#e09b37"),
                                        ("load_bytes", .1, "LOAD link", "#56a88c"),
                                        ("store_bytes", .1, "STORE link", "#b08ec7"))
    for ax, start, end, title in ((axes[1], 0, 3, "M=8, K=16, batch=8"),
                                  (axes[2], 3, 5, "M=8, K=256, batch=8")):
        x = np.arange(end-start)
        bottom = np.zeros(end-start)
        for field, factor, label, color in components:
            heights = np.array([float(v[field])*factor for v in values[start:end]])
            ax.bar(x, heights, bottom=bottom, label=label, color=color, width=.65)
            bottom += heights
        for index, height in enumerate(bottom):
            ax.text(index, height+max(bottom)*.025, f"{height:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x, labels[start:end], fontsize=8)
        ax.set(ylabel="Cold batch time (us)", title=title)
        ax.set_ylim(0, max(bottom)*1.17)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.7, .065),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Tiny-LPU ISA experiments: integer stream, ACC32, sequential execution", fontsize=12)
    fig.text(.5, .015, "Scenario: 100 MHz / 10 MB/s; no host dispatch, quantization, aggregation wait or host-sum cost. Not measured silicon.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(FOLDER/"isa_tradeoffs.png", dpi=180)
    fig.savefig(FOLDER/"isa_tradeoffs.svg")


if __name__ == "__main__":
    main()
