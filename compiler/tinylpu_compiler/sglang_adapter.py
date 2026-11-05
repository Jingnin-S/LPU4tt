"""Opt-in post-load Linear quant_method interposer for CPU offline experiments.

Matches SGLang's apply(layer, x, bias) boundary. It is not an SRT device plugin.
Unsupported calls delegate to the original method; supported calls PTQ to INT4,
run C++ ISA or complete pin/cycle device simulation, then dequantize on host.
No package-wide monkey patching; neither simulator is an SRT device driver.
"""
import hashlib
import time

import numpy as np

from .architecture import ArchRuntime, Resources, choose_plan
from .isa import IsaRuntime, Profile
from .precision import quantization_scale, quantize
from .device import DeviceRuntime


class TinyLpuLinearMethod:
    def __init__(self, original, isa_library, architecture_library, profile=None, resources=None, device_library=None):
        self.original = original
        self.runtime = DeviceRuntime(device_library) if device_library else IsaRuntime(isa_library)
        self.analyzer = ArchRuntime(architecture_library)
        self.placement = 'complete device pin/cycle simulator' if device_library else 'ISA v0.1 simulator'
        self.profile = profile or Profile(memory_byte_bw=1)
        self.resources = resources or Resources()
        self.cache, self.last_debug = {}, {}
        self.offload_calls, self.fallback_calls = 0, 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply(self, layer, x, bias=None):
        import torch
        self.last_debug = {}
        weight = layer.weight
        reason = None
        if x.device.type != "cpu" or weight.device.type != "cpu" or x.dtype != torch.float32 or weight.dtype != torch.float32:
            reason = "requires CPU float32 tensors"
        elif bool(getattr(layer, "use_intel_amx_backend", False)):
            reason = "packed CPU weights require the original method"
        elif x.requires_grad or weight.requires_grad and torch.is_grad_enabled():
            reason = "inference only"
        elif weight.ndim != 2 or x.ndim < 1 or x.shape[-1] != weight.shape[1]:
            reason = "unsupported linear shape"
        else:
            m, k = weight.shape
            frames = x.numel()//k if k else 0
            if not 1 <= m <= 64 or not 1 <= k <= 4096 or not 1 <= frames <= 32:
                reason = "shape exceeds compiler limits"
            elif not torch.isfinite(x).all() or not torch.isfinite(weight).all():
                reason = "nonfinite input or weights"
            elif bias is not None and (not isinstance(bias, torch.Tensor) or
                 bias.device != x.device or bias.dtype != x.dtype or tuple(bias.shape) != (m,)):
                reason = "unsupported bias contract"
        if reason is not None:
            self.fallback_calls += 1
            self.last_debug = {"placement": "original method", "reason": reason}
            return self.original.apply(layer, x, bias)
        start = time.perf_counter_ns()
        a = x.detach().numpy().reshape(frames, k)
        w = weight.detach().numpy()
        digest = hashlib.sha256(w.tobytes()).hexdigest()
        key = (digest, frames, m, k)
        if key not in self.cache:
            ws = quantization_scale(w, 4, per_row=True)
            wq, _ = quantize(w, ws, 4, per_row=True)
            plan, resources, search = choose_plan(wq.tolist(), frames, self.profile, self.resources,
                                                  self.analyzer, "bank4", output_bits=32)
            self.cache[key] = plan, resources, ws
        plan, resources, ws = self.cache[key]
        scales = np.asarray([quantization_scale(row, 4) for row in a], dtype=np.float32)
        aq = np.stack([quantize(row, scale, 4)[0] for row, scale in zip(a, scales)])
        raw, report = self.runtime.run(plan, aq.tolist())
        output = (np.asarray(raw, dtype=np.float64)*scales.astype(np.float64)[:, None]*ws.astype(np.float64)[None, :]).astype(np.float32)
        result = torch.from_numpy(output).reshape(*x.shape[:-1], m)
        if bias is not None:
            result = result+bias
        self.offload_calls += 1
        self.last_debug = {"placement": self.placement, "runtime": report, "resources": resources,
                           "weight_sha256": digest, "activation_scales": scales.tolist(),
                           "weight_scales": ws.tolist(), "raw_integer_output": raw,
                           "host_steps": "dynamic symmetric RNE PTQ; INT4; dequant; optional bias",
                           "wall_ns": time.perf_counter_ns()-start,
                           "accuracy": "approximate versus FP32; exact versus declared quantized arithmetic"}
        return result


def attach_linear(layer, isa_library, architecture_library, **kwargs):
    """Explicitly attach after weight loading; caller retains original for restore."""
    original = layer.quant_method
    if isinstance(original, TinyLpuLinearMethod):
        raise ValueError("Linear already has a Tiny-LPU adapter")
    layer.quant_method = TinyLpuLinearMethod(original, isa_library, architecture_library, **kwargs)
    return original
