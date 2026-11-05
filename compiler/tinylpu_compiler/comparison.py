"""Compare sequential commands with a conservative, memory-aware overlap model.

Only the sequential program has an executable ABI implementation. The overlap
calendar assumes queued asynchronous commands, one half-duplex link (including
setup), one core, one host, and independent storage access for disjoint bytes.
It is a candidate schedule, not hardware/VM DMA or an optimal scheduler.
"""

import heapq

from .performance import PerformanceScenario, estimate_sequence


def compare_schedule(manifest, scenario=None, overlap=False, capture=False):
    scenario = scenario or PerformanceScenario()
    baseline = estimate_sequence([{"kind": "device_plan", "manifest": manifest}], scenario)
    frames, rows = manifest["frames"], manifest["output_rows"]
    if type(overlap) is not bool:
        raise ValueError("overlap must be boolean")
    busy = {"host": sum(baseline["components_ns"][key] for key in
                        ("vm_boundary", "host_dispatch", "host_pack", "host_unpack", "host_quant")),
            "link": sum(baseline["components_ns"][key] for key in
                        ("command_tx", "payload_tx", "payload_rx", "device_setup")),
            "core": baseline["components_ns"]["core"]}
    if not overlap:
        completion = [0.0]*frames
        events = []
        for event in baseline["events"]:
            category, index = event["category"], event["instruction"]
            if category == "host_unpack":
                command = manifest["instructions"][index]
                frame = command["args"][2]//rows
                completion[frame] = max(completion[frame], event["end_ns"])
            resource = "core" if category == "core" else ("link" if category in
                        ("command_tx", "payload_tx", "payload_rx", "device_setup") else "host")
            if capture:
                events.append({**event, "kind": category, "resource": resource})
        result = _result(manifest, baseline, baseline["total_ns"], completion, busy, events, False)
        if capture:
            result["memory_accesses"] = []
        return result

    tasks, last_write, readers, accesses = [], {}, {}, []
    final_outputs = [[] for _ in range(frames)]

    def task(resource, kind, duration, dependencies, index=None, **details):
        number = len(tasks)
        tasks.append({"id": number, "resource": resource, "kind": kind, "duration_ns": duration,
                      "dependencies": sorted(set(dependencies)), "instruction": index, **details})
        return number

    last_dispatch = task("host", "vm_boundary", scenario.vm_offload_call_ns, [])

    def byte_range(address, count):
        if address < 0 or count < 1 or address+count > manifest["target"]["scratchpad_bytes"]:
            raise ValueError("Memory access outside target capacity")
        return range(address, address+count)

    def dependencies(read_ranges, write_ranges):
        deps = set()
        for address, count in read_ranges:
            for byte in byte_range(address, count):
                if byte not in last_write:
                    raise ValueError("Schedule reads uninitialized scratchpad")
                deps.add(last_write[byte])
        for address, count in write_ranges:
            for byte in byte_range(address, count):
                if byte in last_write:
                    deps.add(last_write[byte])
                deps.update(readers.get(byte, ()))
        return deps

    def commit_memory(read_ranges, write_ranges, task_id):
        for mode, ranges in (("read", read_ranges), ("write", write_ranges)):
            for address, count in ranges:
                accesses.append({"task": task_id, "mode": mode, "address": address, "bytes": count})
                for byte in byte_range(address, count):
                    if mode == "read":
                        readers.setdefault(byte, set()).add(task_id)
                    else:
                        last_write[byte] = task_id
                        readers[byte] = set()

    for index, instruction in enumerate(manifest["instructions"]):
        op, args = instruction["op"], instruction["args"]
        dispatch_dependencies = [last_dispatch]
        if op == "LOAD_A":
            pack = task("host", "host_pack", args[2]*scenario.host_pack_ns_per_element,
                        dispatch_dependencies, index)
            dispatch_dependencies = [pack]
        elif op == "END":
            dispatch_dependencies += [event for frame in final_outputs for event in frame]
        last_dispatch = task("host", "host_dispatch", scenario.host_dispatch_ns,
                             dispatch_dependencies, index, opcode=op)
        if op in ("ALLOC_TENSOR", "END"):
            continue
        read_ranges, write_ranges, tx, rx = [], [], 0, 0
        if op == "LOAD_A":
            tx = (args[2]+1)//2
            write_ranges = [(args[3], tx)]
        elif op == "LOAD_W":
            tx = args[1]
            write_ranges = [(args[2], tx)]
        elif op == "GEMV":
            count, k = args[3:5]
            read_ranges = [(args[0], (k+1)//2), (args[1], count*((k+1)//2))]
            write_ranges = [(args[2], count*2)]
        elif op == "STORE_Y":
            rx = args[3]*2
            read_ranges = [(args[0], rx)]
        else:
            raise ValueError("Overlap model supports batch-linear device plans only")
        memory_deps = dependencies(read_ranges, write_ranges)
        # Treat command+setup+payload as one atomic half-duplex transaction.
        # No second command borrows the setup gap or inserts payload bytes.
        link_ns = ((scenario.device_command_bytes+tx)*1e9/scenario.tx_bytes_per_second
                   + rx*1e9/scenario.rx_bytes_per_second
                   + scenario.device_setup_cycles*1e9/scenario.core_hz)
        link = task("link", op, link_ns, [last_dispatch, *memory_deps], index,
                    command_bytes=scenario.device_command_bytes, tx_payload_bytes=tx, rx_payload_bytes=rx)
        memory_end = link
        if op == "GEMV":
            cycles = ((args[3]+manifest["target"]["lanes"]-1)//manifest["target"]["lanes"])*args[4]
            cycles += manifest["target"]["depth"]-1
            memory_end = task("core", "GEMV", cycles*1e9/scenario.core_hz, [link], index,
                              core_cycles=cycles)
        commit_memory(read_ranges, write_ranges, memory_end)
        if op == "STORE_Y":
            frame = args[2]//rows
            if frame >= frames or (args[2]+args[3]-1)//rows != frame:
                raise ValueError("Output store crosses a frame boundary")
            unpack = task("host", "host_unpack", args[3]*scenario.host_unpack_ns_per_element,
                          [link], index, frame=frame)
            final_outputs[frame].append(unpack)

    # Dispatch only ready work. Reserving a future STORE before its result is
    # ready would prevent an independent A prefetch from using the idle link.
    successors = [[] for _ in tasks]
    remaining = [len(item["dependencies"]) for item in tasks]
    ready = {name: [] for name in busy}
    for item in tasks:
        for dependency in item["dependencies"]:
            successors[dependency].append(item["id"])
        if not item["dependencies"]:
            heapq.heappush(ready[item["resource"]], item["id"])
    active, occupied, completed = [], set(), 0
    end_times, now = [0.0]*len(tasks), 0.0

    def finish(index):
        nonlocal completed
        completed += 1
        for successor in successors[index]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                heapq.heappush(ready[tasks[successor]["resource"]], successor)

    while completed < len(tasks):
        while active and active[0][0] <= now:
            _, index = heapq.heappop(active)
            occupied.remove(tasks[index]["resource"])
            finish(index)
        progressed = True
        while progressed:
            progressed = False
            for resource in ("host", "link", "core"):
                if resource in occupied or not ready[resource]:
                    continue
                index = heapq.heappop(ready[resource])
                item = tasks[index]
                item["start_ns"], item["end_ns"] = now, now+item["duration_ns"]
                end_times[index] = item["end_ns"]
                if item["duration_ns"]:
                    occupied.add(resource)
                    heapq.heappush(active, (item["end_ns"], index))
                else:
                    finish(index)
                progressed = True
        if completed == len(tasks):
            break
        if not active:
            raise ValueError("Async task dependency deadlock")
        now = active[0][0]
    if any(not frame for frame in final_outputs):
        raise ValueError("Incomplete batch output")
    completion = [max(end_times[index] for index in frame) for frame in final_outputs]
    total = max(end_times)
    # A greedy asynchronous calendar can choose a worse schedule. Keep an
    # explicit synchronous fallback rather than reporting a fictitious win.
    if total > baseline["total_ns"]+1e-7:
        fallback = compare_schedule(manifest, scenario, overlap=False, capture=capture)
        fallback["overlap_requested"] = True
        fallback["policy"] += "; async greedy candidate slower, synchronous fallback selected"
        return fallback
    result = _result(manifest, baseline, total, completion, busy, tasks if capture else [], True)
    if capture:
        result["memory_accesses"] = accesses
    return result


def _result(manifest, baseline, total, completion, busy, events, overlap):
    frames = manifest["frames"]
    return {"scenario": baseline["scenario"], "total_ns": total,
            "first_frame_ns": completion[0], "mean_frame_completion_ns": sum(completion)/frames,
            "frame_completion_ns": completion, "amortized_frame_ns": total/frames,
            "inferences_per_second": frames*1e9/total,
            "effective_macs_per_second": baseline["counts"]["macs"]*1e9/total,
            "counts": baseline["counts"], "components_ns": baseline["components_ns"],
            "resource_busy_ns": busy, "resource_utilization": {key:value/total for key,value in busy.items()},
            "overlap_saved_ns": max(0.0, baseline["total_ns"]-total),
            "overlap_requested": overlap, "events": events,
            "policy": ("hypothetical queued async calendar; single half-duplex atomic link; one core; one host; byte RAW/WAR/WAW dependencies"
                       if overlap else "executable synchronous ABI-v1 command ordering"),
            "batch_arrival": "all input frames available at t=0; request aggregation wait excluded",
            "physical_performance_validated": False,
            "async_execution_implemented": False}


def validate_calendar(result):
    """Independently check a captured asynchronous calendar's resource/memory hazards."""
    events = result["events"]
    for resource in ("host", "link", "core"):
        ordered = sorted((event for event in events if event["resource"] == resource
                          and event["duration_ns"] > 0), key=lambda event:event["start_ns"])
        for old, new in zip(ordered, ordered[1:]):
            if old["end_ns"] > new["start_ns"]+1e-7:
                raise ValueError("Overlapping events on " + resource)
    if events and "id" in events[0]:
        by_id = {event["id"]:event for event in events}
        for event in events:
            for dependency in event["dependencies"]:
                if by_id[dependency]["end_ns"] > event["start_ns"]+1e-7:
                    raise ValueError("Schedule violates a dependency")
        accesses = result["memory_accesses"]
        for index, left in enumerate(accesses):
            for right in accesses[index+1:]:
                if left["task"] == right["task"] or (left["mode"] == right["mode"] == "read"):
                    continue
                if left["address"] >= right["address"]+right["bytes"] or right["address"] >= left["address"]+left["bytes"]:
                    continue
                a, b = by_id[left["task"]], by_id[right["task"]]
                if min(a["end_ns"], b["end_ns"]) > max(a["start_ns"], b["start_ns"])+1e-7:
                    raise ValueError("Overlapping conflicting scratchpad accesses")
    return True
