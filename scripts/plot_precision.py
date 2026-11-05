#!/usr/bin/env python3
"""Create a standalone numerical-comparison figure from measured experiment rows."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=Path(__file__).resolve().parents[1]/"reports/precision")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((args.report/"results.json").read_text(encoding="utf-8"))
    selected = [("int4_acc16","dynamic_absmax","INT4"),("int8_acc32","dynamic_absmax","INT8"),
                ("fp16_acc32_out32","none","FP16"),("bf16_acc32_out32","none","BF16")]
    groups = [next(group for group in report["summary_groups"] if group["mode"] == mode
                   and group["quantization"] == policy) for mode,policy,_ in selected]
    labels = [label for _,_,label in selected]
    colors = ["#607d8b","#2876a6","#dd9450","#468466"]
    fig,axes = plt.subplots(1,3,figsize=(12.6,4.7))
    for ax,key,title in zip(axes[:2],["normal_median_nrmse","small_median_nrmse"],
                           ["Normal inputs","Tiny inputs (~1e-9)"]):
        values = [group[key] for group in groups]
        ax.bar(labels,values,color=colors,width=.65)
        ax.set_yscale("log")
        ax.set_ylim(1e-5,5)
        ax.set_ylabel("Median NRMSE (lower is better)")
        ax.set_title(title,fontweight="bold",fontsize=11)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        for index,value in enumerate(values):
            ax.text(index,value*1.17,"{:.3g}".format(value),ha="center",fontsize=9)
    values = [group["nonfinite_result_rows"] for group in groups]
    axes[2].bar(labels,values,color=colors,width=.65)
    axes[2].set_ylim(0,49)
    axes[2].set_yticks([0,7,14,28,49])
    axes[2].set_ylabel("Fixtures with NaN/Inf outputs (of 49)")
    axes[2].set_title("Range stress across all fixtures",fontweight="bold",fontsize=11)
    axes[2].grid(axis="y",alpha=.2)
    axes[2].set_axisbelow(True)
    for index,value in enumerate(values):
        axes[2].text(index,value+1,"{}/49".format(value),ha="center",fontsize=9)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Synthetic numerical comparison against a PyTorch FP32 baseline",fontsize=14,fontweight="bold")
    fig.text(.5,.015,"INT4/INT8: dynamic activation + per-channel weight scales. FP16/BF16: FP32 accumulation/output.\n"
             "7 graphs, 7 input distributions, 16 inputs per fixture. These are not trained-model accuracy scores.",
             ha="center",fontsize=9,color="#444444")
    fig.tight_layout(rect=[0,.11,1,.94])
    fig.savefig(args.report/"precision_comparison.png",dpi=180)
    fig.savefig(args.report/"precision_comparison.svg")
    plt.close(fig)
    print(args.report/"precision_comparison.png")


if __name__ == "__main__":
    main()
