"""Numerical experiments, deliberately separate from the device ABI-v1 types."""

import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FORMAT_IDS = {"int4": 0, "int8": 1, "float16": 2, "bfloat16": 3, "float32": 4}
BITS = {"int4": 4, "int8": 8, "int16": 16, "int32": 32,
        "float16": 16, "bfloat16": 16, "float32": 32}


@dataclass(frozen=True)
class NumericMode:
    name: str
    activation_dtype: str
    weight_dtype: str
    accumulator_dtype: str
    output_dtype: str

    def validate(self):
        if self.activation_dtype != self.weight_dtype or self.activation_dtype not in FORMAT_IDS:
            raise ValueError("Reference currently supports equal A/W operand types")
        if self.activation_dtype.startswith("int"):
            if self.accumulator_dtype not in ("int16", "int32") or self.output_dtype != self.accumulator_dtype:
                raise ValueError("Integer reference stores the raw accumulator width")
        elif self.accumulator_dtype != "float32" or self.output_dtype not in ("float32", self.activation_dtype):
            raise ValueError("Floating reference uses FP32 accumulation and explicit output conversion")

    def to_dict(self):
        return {**asdict(self), "implemented_by_abi_v1": self.activation_dtype == "int4"
                and self.accumulator_dtype == "int16" and self.output_dtype == "int16"}


MODES = (
    NumericMode("int4_acc16", "int4", "int4", "int16", "int16"),
    NumericMode("int4_acc32", "int4", "int4", "int32", "int32"),
    NumericMode("int8_acc32", "int8", "int8", "int32", "int32"),
    NumericMode("fp16_acc32_out32", "float16", "float16", "float32", "float32"),
    NumericMode("bf16_acc32_out32", "bfloat16", "bfloat16", "float32", "float32"),
    NumericMode("fp16_acc32_out16", "float16", "float16", "float32", "float16"),
    NumericMode("bf16_acc32_out16", "bfloat16", "bfloat16", "float32", "bfloat16"),
    NumericMode("fp32_ordered", "float32", "float32", "float32", "float32"),
)


def quantization_scale(calibration, bits, per_row=False):
    values = np.asarray(calibration, dtype=np.float32)
    if bits not in (4, 8) or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Quantization requires finite nonempty data and INT4/INT8")
    if per_row and values.ndim != 2:
        raise ValueError("Per-row calibration requires a matrix")
    maximum = np.max(np.abs(values.astype(np.float64)), axis=1 if per_row else None)
    scale = np.asarray(maximum / ((1 << (bits-1))-1), dtype=np.float32)
    return np.where(scale > 0, scale, np.float32(1)).astype(np.float32)


def quantize(values, scale, bits, per_row=False):
    values = np.asarray(values, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if bits not in (4,8) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid quantization data or scale")
    if per_row and (values.ndim != 2 or scale.ndim != 1 or scale.size != values.shape[0]):
        raise ValueError("Per-row scales must match matrix rows")
    divisor = scale[:,None] if per_row else scale
    normalized = values.astype(np.float64) / divisor.astype(np.float64)
    rounded = np.rint(normalized)  # round to nearest, ties to even
    minimum, maximum = -(1 << (bits-1)), (1 << (bits-1))-1
    clipped = (rounded < minimum) | (rounded > maximum)
    return np.clip(rounded, minimum, maximum).astype(np.int32), int(np.count_nonzero(clipped))


def torch_convert(values, dtype):
    """Independent conversion oracle; NaN payloads are not part of the contract."""
    import torch

    values = np.ascontiguousarray(values, dtype=np.float32)
    converted = torch.from_numpy(values).to(getattr(torch, dtype))
    decoded = converted.to(torch.float32).numpy().copy()
    encoded = None if dtype == "float32" else converted.view(torch.int16).numpy().view(np.uint16).copy()
    return decoded, encoded


def assert_same_float32(actual, expected, label="numeric reference"):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise RuntimeError(label + " shape mismatch")
    nan = np.isnan(actual) & np.isnan(expected)
    if not np.all((actual.view(np.uint32) == expected.view(np.uint32)) | nan):
        raise RuntimeError(label + " bit/class mismatch")


class PrecisionRuntime:
    """C++ numeric reference binding; no target ISA capability is implied."""

    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        fptr, dptr = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
        uptr = ctypes.POINTER(ctypes.c_uint64)
        self.library.tlpu_precision_gemv.argtypes = [ctypes.c_int, ctypes.c_int, fptr, fptr,
                                                   ctypes.c_size_t, ctypes.c_size_t, dptr, uptr, uptr, uptr]
        self.library.tlpu_precision_gemv.restype = ctypes.c_int
        self.library.tlpu_precision_convert.argtypes = [ctypes.c_int, fptr, ctypes.c_size_t, fptr,
                                                       ctypes.POINTER(ctypes.c_uint16)]
        self.library.tlpu_precision_convert.restype = ctypes.c_int
        self.library.tlpu_precision_last_error.argtypes = []
        self.library.tlpu_precision_last_error.restype = ctypes.c_char_p

    def _check(self, status):
        if status:
            error = self.library.tlpu_precision_last_error()
            raise RuntimeError(error.decode("utf-8") if error else "C++ precision reference failed")

    def convert(self, values, dtype):
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("Conversion API accepts FP16 or BF16")
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not values.size:
            raise ValueError("Conversion input must be nonempty")
        output = np.empty_like(values)
        encoded = np.empty(values.shape, dtype=np.uint16)
        fptr = ctypes.POINTER(ctypes.c_float)
        self._check(self.library.tlpu_precision_convert(FORMAT_IDS[dtype],
                    values.ctypes.data_as(fptr), values.size, output.ctypes.data_as(fptr),
                    encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
        return output, encoded

    def gemv(self, a, w, mode):
        mode.validate()
        a, w = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(w, dtype=np.float32)
        if a.ndim != 1 or w.ndim != 2 or w.shape[1] != a.size:
            raise ValueError("GEMV reference expects [K] and [rows,K]")
        output = np.empty(w.shape[0], dtype=np.float64)
        overflows, products, accumulators = (ctypes.c_uint64() for _ in range(3))
        fptr = ctypes.POINTER(ctypes.c_float)
        self._check(self.library.tlpu_precision_gemv(FORMAT_IDS[mode.activation_dtype],
                    BITS[mode.accumulator_dtype], a.ctypes.data_as(fptr), w.ctypes.data_as(fptr),
                    w.shape[0], a.size, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.byref(overflows), ctypes.byref(products), ctypes.byref(accumulators)))
        return output, {"integer_overflow_events": overflows.value,
                        "nonfinite_product_events": products.value,
                        "nonfinite_accumulator_events": accumulators.value}


def ordered_reference(a, w, mode):
    """Independent scalar oracle, with Python integers or explicit np.float32 ops."""
    mode.validate()
    a, w = np.asarray(a, dtype=np.float32), np.asarray(w, dtype=np.float32)
    counters = {"integer_overflow_events": 0, "nonfinite_product_events": 0,
                "nonfinite_accumulator_events": 0}
    output = []
    if mode.activation_dtype.startswith("int"):
        bits = BITS[mode.activation_dtype]
        low, high = -(1 << (bits-1)), (1 << (bits-1))-1
        for value in (a, w):
            if not np.all(np.isfinite(value)) or not np.all(value == np.rint(value)) or np.any(value < low) or np.any(value > high):
                raise ValueError("Raw integer operand is outside declared type")
        width = BITS[mode.accumulator_dtype]
        half, modulus = 1 << (width-1), 1 << width
        for row in w:
            accumulator = 0
            for x,y in zip(a,row):
                total = accumulator + int(x)*int(y)
                counters["integer_overflow_events"] += int(total < -half or total >= half)
                accumulator = (total+half) % modulus - half
            output.append(accumulator)
    else:
        a, _ = torch_convert(a, mode.activation_dtype)
        w, _ = torch_convert(w, mode.weight_dtype)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for row in w:
                accumulator = np.float32(0)
                for x,y in zip(a,row):
                    product = np.float32(x*y)
                    accumulator = np.float32(accumulator+product)
                    counters["nonfinite_product_events"] += int(not np.isfinite(product))
                    counters["nonfinite_accumulator_events"] += int(not np.isfinite(accumulator))
                output.append(float(accumulator))
    return np.asarray(output, dtype=np.float64), counters


def error_metrics(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError("Error metric shape mismatch")
    if not reference.size:
        raise ValueError("Error metrics require nonempty arrays")
    mask = np.isfinite(reference) & np.isfinite(actual)
    result = {"elements": actual.size, "reference_nonfinite": int(np.count_nonzero(~np.isfinite(reference))),
              "nonfinite_outputs": int(np.count_nonzero(~np.isfinite(actual))),
              "nan_outputs": int(np.count_nonzero(np.isnan(actual))),
              "positive_inf_outputs": int(np.count_nonzero(np.isposinf(actual))),
              "negative_inf_outputs": int(np.count_nonzero(np.isneginf(actual)))}

    def finite_metrics(ref, value):
        if not ref.size:
            return None
        error = value-ref
        denominator, prediction_norm = np.linalg.norm(ref), np.linalg.norm(value)
        cosine = float(np.dot(ref.reshape(-1),value.reshape(-1))/(denominator*prediction_norm)) if denominator and prediction_norm else float(not denominator and not prediction_norm)
        return {"max_abs_error": float(np.max(np.abs(error))), "mean_abs_error": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error*error))),
                "nrmse": float(np.linalg.norm(error)/denominator) if denominator else (0.0 if not np.any(error) else None),
                "cosine": float(np.clip(cosine,-1,1))}

    result["finite_subset"] = finite_metrics(reference[mask],actual[mask])
    result["complete_finite"] = bool(np.all(mask))
    result.update(result["finite_subset"] if result["complete_finite"] else
                  {key:None for key in ("max_abs_error", "mean_abs_error", "rmse", "nrmse", "cosine")})
    return result


def required_accumulator_bits(a_bits, w_bits, k):
    if a_bits not in (4,8) or w_bits not in (4,8) or type(k) is not int or k < 1:
        raise ValueError("Invalid signed integer dot range")
    maximum = k*(1 << (a_bits-1))*(1 << (w_bits-1))
    negative = k*max((1 << (a_bits-1))*((1 << (w_bits-1))-1),
                     ((1 << (a_bits-1))-1)*(1 << (w_bits-1)))
    bits = 1
    while maximum > (1 << (bits-1))-1 or negative > (1 << (bits-1)):
        bits += 1
    return {"required_signed_bits": bits, "maximum": maximum, "minimum": -negative}


def storage_requirement(rows, k, mode, buffers=2, lanes=2, quantization="dynamic_absmax"):
    mode.validate()
    if min(rows,k) < 1 or buffers not in (1,2) or lanes not in (1,2,4):
        raise ValueError("Invalid storage geometry")
    row_bytes = (k*BITS[mode.weight_dtype]+7)//8
    weight_bytes = ((rows+lanes-1)//lanes)*lanes*row_bytes
    activation_bytes = (k*BITS[mode.activation_dtype]+7)//8
    output_element_bytes = BITS[mode.output_dtype]//8
    position = weight_bytes
    for _ in range(buffers):
        position += activation_bytes
        position = (position+output_element_bytes-1)//output_element_bytes*output_element_bytes
        position += rows*output_element_bytes
    quantized = mode.activation_dtype.startswith("int")
    return {"weight_payload_bytes": rows*row_bytes, "weight_bank_reserved_bytes": weight_bytes,
            "a_payload_bytes_per_frame": activation_bytes, "y_payload_bytes_per_frame": rows*output_element_bytes,
            "data_scratchpad_reserved_bytes": position,
            "host_quantization_parameter_bytes": (rows+(buffers if quantization == "dynamic_absmax" else 1))*4 if quantized else 0,
            "accumulator_register_bytes": lanes*BITS[mode.accumulator_dtype]//8,
            "metadata_location": "host; no on-device scaling is implied",
            "scope": "full kernel resident geometry, no tiling or RTL area estimate"}
