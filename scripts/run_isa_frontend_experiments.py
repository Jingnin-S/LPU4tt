#!/usr/bin/env python3
"""Real torch.export -> Relax partition -> Tiny IR -> new ISA -> C++ execution."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"compiler"), str(ROOT/"scripts")]
from run_compiler_experiments import build_models
from tinylpu_compiler.frontend import import_model
from tinylpu_compiler.isa import IsaRuntime
from tinylpu_compiler.isa_graph import IsaGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", default=str(ROOT/"reports/isa/frontend"))
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    runtime = IsaRuntime(args.library)
    rng = np.random.default_rng(20261103)

    class LongLinear(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.register_buffer("w", torch.tensor([[((i*3+r*7) % 5)-2 for i in range(k)] for r in range(3)], dtype=torch.float32))
        def forward(self, x):
            return torch.nn.functional.linear(x, self.w)

    models = build_models(torch)+[("linear_65x3", LongLinear(65), 65), ("linear_256x3", LongLinear(256), 256)]
    records = []
    for name, model, k in models:
        folder = root/name
        folder.mkdir(parents=True, exist_ok=True)
        node, metadata = import_model(model, (torch.zeros(1, k),), folder)
        inputs = rng.integers(-8, 8, size=(16, k), dtype=np.int32)
        with torch.no_grad():
            expected = model(torch.tensor(inputs, dtype=torch.float32)).to(torch.int32).tolist()
        for encoding in (4, 8):
            program = IsaGraph(node, folder/f"encoding_{encoding}", frames=8, encoding=encoding)
            samples = []
            for offset in (0, 8):
                actual, reports, host = program.run(runtime, inputs[offset:offset+8].tolist())
                if actual != expected[offset:offset+8]:
                    raise AssertionError(f"New ISA frontend numerical mismatch: {name}")
                samples.append({"input": inputs[offset:offset+8].tolist(), "result": actual,
                                "reports": reports, "host_quant_elements": host})
            record = {"name": name, "k": k, "encoding": encoding, "samples": 16,
                      "device_partitions": len(program.plans), "byoc_partitions": len(metadata["byoc_functions"]),
                      "all_numerical_checks_passed": True}
            records.append(record)
            (folder/f"results_e{encoding}.json").write_text(json.dumps({"summary": record, "samples": samples}, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"model": name, "new_isa_frontend": "PASS"}), flush=True)
    summary = {"models": 6, "encoding_variants": 12, "runtime_inputs": 192, "records": records,
               "scope": "Real Relax partition frontend and explicit host interpreter; no new TVM VM backend or SGLang service"}
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
