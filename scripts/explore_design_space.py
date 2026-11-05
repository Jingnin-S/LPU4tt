#!/usr/bin/env python3
"""Finite-storage Tiny-LPU ESL design-space analysis; frequencies are targets."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from analyze_esl import load_profiles


def core_cycles(rows, k, lanes, depth, row_drain=False):
    groups = (rows + lanes - 1) // lanes
    return groups * (k + depth - 1) if row_drain else groups * k + depth - 1


def storage_bytes(rows, k, lanes, buffers):
    """Bank padding is allocated even when the last lane is masked."""
    weight_rows = ((rows + lanes - 1) // lanes) * lanes
    return weight_rows * ((k + 1) // 2) + buffers * ((k + 1) // 2 + 2 * rows)


def make_tiles(workload, arch, capacity):
    m, k = workload["rows"], workload["k"]
    legal = [r for r in range(1, m + 1)
             if storage_bytes(r, k, arch["lanes"], arch["buffers"]) <= capacity]
    if not legal:
        return []
    size = max(legal)
    return [min(size, m - first) for first in range(0, m, size)]


def schedule(workload, arch, capacity, frequency_mhz, bandwidth_MBps,
             order="weight_stationary", command_bytes=1, capture=False):
    """One non-preemptive half-duplex link, one engine, finite A/Y slots.

    Completed outputs have IO priority, then the next ordered input is loaded.
    A weight-bank switch waits until all readers of the old tile have finished.
    This is an executable policy, not a claim of globally optimal overlap.
    """
    if frequency_mhz <= 0 or bandwidth_MBps <= 0 or capacity <= 0:
        raise ValueError("Frequency, bandwidth and capacity must be positive")
    if order not in ("weight_stationary", "latency_first"):
        raise ValueError("Unknown order")
    tiles = make_tiles(workload, arch, capacity)
    if not tiles:
        raise ValueError("Insufficient scratchpad capacity")
    frames, k, lanes = workload["frames"], workload["k"], arch["lanes"]
    pairs = ([(t, f) for t in range(len(tiles)) for f in range(frames)]
             if order == "weight_stationary" else
             [(t, f) for f in range(frames) for t in range(len(tiles))])
    jobs = [{"tile": t, "frame": f, "rows": tiles[t], "state": "pending", "slot": None}
            for t, f in pairs]
    slots = [None] * arch["buffers"]
    now, next_input, completed, resident = 0, 0, 0, None
    active_io = active_compute = None
    io_ticks = compute_ticks = transferred = weight_bytes = 0
    frame_finish = [0] * frames
    trace = []

    def transfer_ticks(byte_count):
        return math.ceil(byte_count * frequency_mhz / bandwidth_MBps)

    def record(kind, job, start, end, byte_count=0):
        if capture:
            trace.append({"resource": "compute" if kind == "compute" else "io",
                          "kind": kind, "job": job, "tile": jobs[job]["tile"],
                          "frame": jobs[job]["frame"], "slot": jobs[job]["slot"],
                          "start_tick": start, "end_tick": end, "bytes": byte_count})

    while completed < len(jobs):
        if active_io is not None and active_io[0] == now:
            _, kind, index = active_io
            job = jobs[index]
            if kind == "weight":
                resident = job["tile"]
            elif kind == "input":
                job["state"] = "ready"
            else:
                job["state"] = "done"
                slots[job["slot"]] = None
                completed += 1
                frame_finish[job["frame"]] = max(frame_finish[job["frame"]], now)
            active_io = None
        if active_compute is not None and active_compute[0] == now:
            jobs[active_compute[1]]["state"] = "output"
            active_compute = None
        if completed == len(jobs):
            break

        if active_compute is None:
            ready = next((i for i, j in enumerate(jobs) if j["state"] == "ready"), None)
            if ready is not None:
                job = jobs[ready]
                if job["tile"] != resident:
                    raise RuntimeError("Weight bank changed while an input was waiting")
                ticks = core_cycles(job["rows"], k, lanes, arch["depth"], arch["row_drain"])
                job["state"] = "running"
                active_compute = (now + ticks, ready)
                compute_ticks += ticks
                record("compute", ready, now, now + ticks)

        if active_io is None:
            output = next((i for i, j in enumerate(jobs) if j["state"] == "output"), None)
            kind, index, byte_count = None, None, 0
            if output is not None:
                index, kind = output, "output"
                jobs[index]["state"] = "reading"
                byte_count = 2 * jobs[index]["rows"]
            elif next_input < len(jobs) and None in slots:
                index = next_input
                job = jobs[index]
                if resident != job["tile"]:
                    if not any(j["state"] in ("ready", "running") for j in jobs):
                        kind = "weight"
                        byte_count = job["rows"] * ((k + 1) // 2)
                        weight_bytes += byte_count
                else:
                    kind = "input"
                    job["slot"] = slots.index(None)
                    slots[job["slot"]] = index
                    job["state"] = "loading"
                    next_input += 1
                    byte_count = (k + 1) // 2 + command_bytes
            if kind is not None:
                ticks = transfer_ticks(byte_count)
                active_io = (now + ticks, kind, index)
                io_ticks += ticks
                transferred += byte_count
                record(kind, index, now, now + ticks, byte_count)

        ends = [e[0] for e in (active_io, active_compute) if e is not None]
        if not ends or min(ends) <= now:
            raise RuntimeError("Scheduler deadlock or zero-duration event")
        now = min(ends)

    macs = workload["rows"] * k * frames
    data_bytes = max(storage_bytes(r, k, lanes, arch["buffers"]) for r in tiles)
    tag_bits = max(1, (workload["rows"] - 1).bit_length()) + max(1, (k - 1).bit_length()) + 3
    pipeline_proxy = lanes * 16 + (arch["depth"] - 1) * lanes * (8 + tag_bits)
    if io_ticks + compute_ticks < now:
        raise RuntimeError("Unaccounted idle time")
    return {
        "tiles": len(tiles), "tile_rows": tiles, "jobs": len(jobs),
        "storage_bytes": data_bytes, "scratchpad_capacity_bytes": capacity,
        "pipeline_state_proxy_bits": pipeline_proxy,
        "total_state_proxy_bits": data_bytes * 8 + pipeline_proxy,
        "macs": macs, "compute_cycles": compute_ticks, "io_cycles": io_ticks,
        "total_cycles": now, "total_us": now / frequency_mhz,
        "first_frame_us": frame_finish[0] / frequency_mhz,
        "mean_frame_completion_us": sum(frame_finish) / frames / frequency_mhz,
        "frame_completion_ticks": frame_finish,
        "effective_mmacs_per_second": macs * frequency_mhz / now,
        "core_mmacs_per_second": macs * frequency_mhz / compute_ticks,
        "peak_mmacs_per_second": lanes * frequency_mhz,
        "compute_mac_utilization": macs / (lanes * compute_ticks),
        "engine_occupancy": compute_ticks / now, "link_occupancy": io_ticks / now,
        "overlap_cycles": io_ticks + compute_ticks - now,
        "transferred_bytes": transferred, "weight_transfer_bytes": weight_bytes,
        "macs_per_transferred_byte": macs / transferred,
        "trace": trace,
    }


def validate_core(model, shapes, output):
    calibration = []
    for rows, k, lanes, depth, drain in sorted(shapes):
        command = [str(model), "--rows", str(rows), "--k", str(k), "--lanes", str(lanes),
                   "--depth", str(depth), "--row-drain", str(int(drain)), "--seed", "1"]
        run = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, timeout=30)
        if run.returncode:
            raise RuntimeError("Core model failed: " + run.stderr[:1000])
        result = json.loads(run.stdout)
        expected = [sum(result["a"][i] * result["w"][r * k + i] for i in range(k)) for r in range(rows)]
        if result["functional_result"] != expected or result["cycle_result"] != expected:
            raise RuntimeError("Core functional mismatch")
        if result["cycles"] != core_cycles(rows, k, lanes, depth, drain):
            raise RuntimeError("Core latency formula mismatch")
        if result["issue_events"] != rows * k or result["commit_events"] != rows * k:
            raise RuntimeError("Core event-count mismatch")
        if (result["rows"], result["k"], result["lanes"], result["depth"], result["row_drain"]) != (rows, k, lanes, depth, drain):
            raise RuntimeError("Core configuration mismatch")
        calibration.append(result)
    write_json(output / "core_calibration.json", {"model": str(model), "cases": calibration})
    return len(calibration)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows)


def dominates(a, b):
    av = (-a["effective_mmacs_per_second"], a["first_frame_us"], a["storage_bytes"],
          a["lanes"], a["pipeline_state_proxy_bits"])
    bv = (-b["effective_mmacs_per_second"], b["first_frame_us"], b["storage_bytes"],
          b["lanes"], b["pipeline_state_proxy_bits"])
    return all(x <= y + 1e-10 for x, y in zip(av, bv)) and any(x < y - 1e-10 for x, y in zip(av, bv))


def report(rows, pareto, output, calibration_count, config):
    lines = ["# Tiny-LPU 高频率与存储设计空间分析", "",
             "核心主目标 **100 MHz**；150/200 MHz 为探索点。所有频率都是假设/目标，尚无综合或 STA；host payload 带宽独立于核心频率。",
             "SKY/IHP/GF 各自绑定固定模板配置，同频同架构的 ESL 性能一致；本轮不推断跨 PDK 的真实 Fmax 或面积。", "",
             "## 建模范围", "",
             "- INT4/ACC16；P=1/2/4 沿输出行并行，A 广播，W 分 bank；流水深度 2/3/4。",
             "- scratchpad=16/32/64/128/256 B，容量包含 bank padding、权重、A 和结果；双缓冲复制 A/结果，不复制权重。",
             "- 有限容量按输出行分块；tile-major 批次内复用权重，frame-major 优先单样本完成。",
             "- 一条半双工 host 链路，1/5/25 MB/s；每 tile/frame 有一次 1-byte START。没有额外地址帧，继承 sideband 假设。",
             "- 离散事件执行 load/input/compute/output，完成输出优先占用 IO；buffer 必须读完才能复用，权重切换等待旧计算结束。",
             "- 对照 v1 逐组排空与跨行连续发射；流水标签携带 first/last，累加在提交时切换行。",
             "- FIR8 使用独立输入窗口，没有免费滑窗复用；批次中 A 为不同请求，时延从批次开始计。", "",
             "## 100 MHz core / 25 MB/s host：各算例吞吐最高的配置", "",
             "以下先比较吞吐，接近时选更小存储/较少 lanes；不代表这些容量能放入所选 tile，也不证明各架构都能达到相同目标频率。", "",
             "| 算例 | 架构 | 数据存储 B | 分块 | 遍历 | MMAC/s | 首样本 µs | MAC利用率 |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    common = [r for r in rows if r["template"] == "sky" and r["core_mhz"] == 100 and r["io_MBps"] == 25]
    selections = []
    for workload in config["workloads"]:
        group = [r for r in common if r["workload"] == workload["id"]]
        best = min(group, key=lambda r: (-round(r["effective_mmacs_per_second"], 9), r["storage_bytes"], r["lanes"], r["pipeline_state_proxy_bits"], r["first_frame_us"]))
        selections.append(best)
        lines.append("| {workload} | {architecture} | {storage_bytes} | {tiles} | {order} | {effective_mmacs_per_second:.3f} | {first_frame_us:.3f} | {compute_mac_utilization:.1%} |".format(**best))
    lines.extend(["", "## 提高频率的收益与链路上限", "",
                  "固定 stream2_ping / 128 B / 25 MB/s / weight-stationary，频率改变不提高外部带宽：", "",
                  "| 算例 | 50 MHz MMAC/s | 100 MHz | 150 MHz | 200 MHz | 100→200 增益 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for workload in config["workloads"]:
        found = {r["core_mhz"]: r for r in rows if r["template"] == "sky" and r["workload"] == workload["id"]
                 and r["architecture"] == "stream2_ping" and r["scratchpad_capacity_bytes"] == 128
                 and r["io_MBps"] == 25 and r["order"] == "weight_stationary"}
        if len(found) == 4:
            v = [found[f]["effective_mmacs_per_second"] for f in (50, 100, 150, 200)]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1%} |".format(workload["id"], *v, v[-1] / v[1] - 1))
    lines.extend(["", "## PE 数与算例匹配", "",
                  "100 MHz / 25 MB/s、3级流水、双缓冲；相同容量预算比较。吞吐包含搬运。", "",
                  "| 算例 / 容量预算 | 1 lane MMAC/s | 2 lanes | 4 lanes | 2→4 收益 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for workload, capacity in (("fir8_b16", 32), ("fc4x16_b16", 64), ("fc8x32_b16", 256)):
        found = {r["lanes"]: r for r in common if r["workload"] == workload
                 and r["architecture"] in ("stream1_ping", "stream2_ping", "stream4_ping")
                 and r["scratchpad_capacity_bytes"] == capacity and r["order"] == "weight_stationary"}
        v = [found[p]["effective_mmacs_per_second"] for p in (1, 2, 4)]
        lines.append("| {} / {} B | {:.3f} | {:.3f} | {:.3f} | {:.2%} |".format(workload, capacity, *v, v[2] / v[1] - 1))
    lines.extend(["", "## 存储容量与批次响应的 trade off", "",
                  "FC8×32、16帧、2 lanes、3级流水、双缓冲，100 MHz / 25 MB/s：", "",
                  "| 容量预算 B | 实用数据 B | 分块 | 遍历 | 搬运 B | MMAC/s | 首样本 µs |",
                  "| ---: | ---: | ---: | --- | ---: | ---: | ---: |"])
    for r in common:
        if r["workload"] == "fc8x32_b16" and r["architecture"] == "stream2_ping" and r["scratchpad_capacity_bytes"] in (128, 256):
            lines.append("| {scratchpad_capacity_bytes} | {storage_bytes} | {tiles} | {order} | {transferred_bytes} | {effective_mmacs_per_second:.3f} | {first_frame_us:.3f} |".format(**r))
    lines.extend(["", "## 建议的下一阶段候选", "",
                  "1. **小面积候选**：1 lane、3级流水、16–32 B 数据容量，优先 dot/GEMV/FIR。FIR8 只有一个输出行，增加行并行 lanes 没有吞吐收益；窗口复用应另做结构验证。",
                  "2. **主线候选**：2 lanes、3级流水、64 B 数据容量和 A/结果双缓冲，面向 FC4×16 的重复推理。100 MHz、25 MB/s 假设下约 84.07 MMAC/s，4 lanes 只再提高约 0.16%，优先保留面积与布线余量。",
                  "3. **较大存储候选**：2 lanes、3级流水、192 B 实用数据（扫描容量档为256 B），完整驻留 FC8×32。相比128 B预算的分块配置，减少输入重传并改善首样本完成时间；是否流片可行由多tile/宏存储与物理报告决定。",
                  "4. **频率取舍**：主目标100 MHz；先验证bank读→乘法→累加及一周期ACC反馈。当前链路条件下100→200 MHz端到端增益有限，150/200 MHz作为时序探索，不优先以面积换取无明显端到端收益的深流水。", ""])
    lines.extend(["", "## 指标与限制", "",
                  "- MMAC/s 来自整个批次的完成时间，含实际装载、控制字节、计算和读回；MAC 计一次乘加。",
                  "- 首样本时间是从批次开始到 frame0 全部分块结果读回；tile-major 可能延后首个完整样本。",
                  "- compute MAC利用率包含填充/排空和尾部空 lanes；engine/link occupancy 描述资源在总时间内占用。",
                  "- `pipeline_state_proxy_bits` 只是累加器/流水标签等状态的估算代理，不能换算成跨工艺硅面积或功耗。数据存储容量单独精确计数。",
                  "- 100 MHz 的 16-bit 累加反馈、bank/mux、扇出和 CTS 仍需真实 STA；加深前级流水不自动证明这些路径更快。",
                  "- SKY130 官方 clock 经输入 pad，额定 66 MHz；更高核心 STA 目标与额定外部可供时钟分开。IHP/GF 的板级条件需分别核查。",
                  "- 128/256 B 是资源压力点，不能直接作为单 tile 的配置承诺；SRAM 宏不是跨三模板默认可用资源。", "",
                  "## 数据与验证", "",
                  "- {} 行可行配置；{} 种 core shape 与真实 C++ 功能/周期模型对照。".format(len(rows), calibration_count),
                  "- [全部 CSV](summary.csv)、[100 MHz / 25 MB/s 多目标 Pareto](pareto.csv)、[core 校准](core_calibration.json)、[事件 trace](representative_traces.json)。",
                  "- 每模板独立目录：" + "、".join("[{}]({}/performance.csv)".format(t.upper(), t) for t in ("sky", "ihp", "gf")),
                  "- [GPIO 官方限制](https://tinytapeout.com/specs/gpio/)、[存储官方说明](https://tinytapeout.com/specs/memory/)。", ""])
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "recommendations.json", selections)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "reports/dse"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs/dse.json").read_text(encoding="utf-8"))
    profiles = load_profiles(root)
    shapes, base, infeasible = set(), [], []
    for workload in config["workloads"]:
        for arch in config["architectures"]:
            for capacity in config["scratchpad_capacity_sweep_bytes"]:
                tiles = make_tiles(workload, arch, capacity)
                if not tiles:
                    infeasible.append({"workload": workload["id"], "architecture": arch["id"], "capacity_bytes": capacity,
                                       "minimum_required_bytes": storage_bytes(1, workload["k"], arch["lanes"], arch["buffers"])})
                    continue
                shapes.update((r, workload["k"], arch["lanes"], arch["depth"], arch["row_drain"]) for r in tiles)
                for frequency in config["core_frequency_sweep_mhz"]:
                    for bandwidth in config["host_payload_bandwidth_sweep_MBps"]:
                        for order in config["orders"]:
                            metrics = schedule(workload, arch, capacity, frequency, bandwidth, order,
                                               config["start_command_bytes"])
                            metrics.pop("trace")
                            base.append(dict(workload=workload["id"], rows=workload["rows"], k=workload["k"], frames=workload["frames"],
                                             architecture=arch["id"], lanes=arch["lanes"], depth=arch["depth"], buffers=arch["buffers"],
                                             row_drain=arch["row_drain"], core_mhz=frequency, io_MBps=bandwidth, order=order, **metrics))
    count = validate_core(args.model.resolve(), shapes, output)
    rows = []
    for profile in profiles:
        specific = [dict(template=profile["id"], pdk=profile["pdk"], template_commit=profile["template_commit"],
                         core_frequency_is_target=True, physical_performance_validated=False, **r) for r in base]
        folder = output / profile["id"]
        folder.mkdir(exist_ok=True)
        write_csv(folder / "performance.csv", specific)
        write_json(folder / "profile.json", profile)
        rows.extend(specific)
    write_csv(output / "summary.csv", rows)
    pareto = []
    for workload in config["workloads"]:
        group = [r for r in rows if r["template"] == "sky" and r["workload"] == workload["id"]
                 and r["core_mhz"] == 100 and r["io_MBps"] == 25]
        unique = {}
        for r in group:
            key = (r["architecture"], tuple(r["tile_rows"]), r["order"])
            unique.setdefault(key, r)
        group = list(unique.values())
        pareto.extend(r for r in group if not any(dominates(other, r) for other in group))
    write_csv(output / "pareto.csv", pareto)
    write_json(output / "infeasible.json", infeasible)
    write_json(output / "run.json", {"config": config, "model_executable": str(args.model.resolve()),
                                     "rows": len(rows), "calibration_cases": count, "physical_performance_validated": False,
                                     "pareto_template_equivalence": "Common core: SKY Pareto applies to all at identical assumptions"})
    report(rows, pareto, output, count, config)
    selected = json.loads((output / "recommendations.json").read_text(encoding="utf-8"))
    traces = []
    for r in selected:
        workload = next(w for w in config["workloads"] if w["id"] == r["workload"])
        arch = next(a for a in config["architectures"] if a["id"] == r["architecture"])
        result = schedule(workload, arch, r["scratchpad_capacity_bytes"], r["core_mhz"], r["io_MBps"], r["order"], capture=True)
        traces.append({"configuration": r, "events": result["trace"]})
    write_json(output / "representative_traces.json", traces)
    print("DSE passed: {} rows, {} calibrated core shapes, {} Pareto candidates; {}".format(len(rows), count, len(pareto), output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print("DSE failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
