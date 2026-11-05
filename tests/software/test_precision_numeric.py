import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))

from tinylpu_compiler.precision import (  # noqa: E402
    BITS,
    MODES,
    NumericMode,
    PrecisionRuntime,
    assert_same_float32,
    error_metrics,
    ordered_reference,
    quantization_scale,
    quantize,
    required_accumulator_bits,
    storage_requirement,
    torch_convert,
)


LIBRARY = Path(os.environ.get(
    "TINYLPU_PRECISION_LIBRARY", ROOT / "build_esl/libtinylpu_precision.so"
))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class PrecisionHelperTests(unittest.TestCase):
    def test_quantization_scale_zero_and_per_row(self):
        zeros = np.zeros((2, 3), dtype=np.float32)
        np.testing.assert_array_equal(quantization_scale(zeros, 4), np.float32(1))
        np.testing.assert_array_equal(
            quantization_scale(zeros, 4, per_row=True), np.ones(2, dtype=np.float32)
        )
        calibration = np.asarray([[0.0, 7.0], [-8.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            quantization_scale(calibration, 4, per_row=True),
            np.asarray([1.0, 8.0 / 7.0], dtype=np.float32),
        )

    def test_quantize_rne_ties_and_clipping(self):
        values = np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
        actual, clipped = quantize(values, np.float32(1), 4)
        np.testing.assert_array_equal(actual, np.asarray([-2, -2, 0, 0, 2, 2]))
        self.assertEqual(clipped, 0)

        clipped_values, clipped_count = quantize(
            np.asarray([-20.0, 0.0, 20.0], dtype=np.float32), np.float32(1), 4
        )
        np.testing.assert_array_equal(clipped_values, np.asarray([-8, 0, 7]))
        self.assertEqual(clipped_count, 2)

        matrix = np.asarray([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float32)
        matrix_q, matrix_clipped = quantize(
            matrix, np.asarray([1.0, 1.0], dtype=np.float32), 4, per_row=True
        )
        np.testing.assert_array_equal(matrix_q, np.asarray([[0, 2], [0, -2]]))
        self.assertEqual(matrix_clipped, 0)

    def test_quantization_rejects_invalid_and_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            quantization_scale(np.asarray([], dtype=np.float32), 4)
        with self.assertRaises(ValueError):
            quantization_scale(np.asarray([np.inf], dtype=np.float32), 4)
        with self.assertRaises(ValueError):
            quantization_scale(np.ones((2, 2), dtype=np.float32), 3)
        with self.assertRaises(ValueError):
            quantization_scale(np.ones(4, dtype=np.float32), 4, per_row=True)
        with self.assertRaises(ValueError):
            quantize(np.asarray([1.0], dtype=np.float32), 0.0, 4)
        with self.assertRaises(ValueError):
            quantize(np.asarray([np.nan], dtype=np.float32), 1.0, 4)
        with self.assertRaises(ValueError):
            quantize(np.ones((2, 2), dtype=np.float32), np.ones(1), 4, per_row=True)
        with self.assertRaises(ValueError):
            quantize(np.ones((2, 2), dtype=np.float32), np.ones(3), 4, per_row=True)

    def test_required_accumulator_bits_boundaries(self):
        self.assertEqual(required_accumulator_bits(4, 4, 64)["required_signed_bits"], 14)
        self.assertEqual(required_accumulator_bits(4, 4, 256)["required_signed_bits"], 16)
        self.assertEqual(required_accumulator_bits(4, 4, 512)["required_signed_bits"], 17)
        self.assertEqual(required_accumulator_bits(4, 4, 64)["maximum"], 4096)
        self.assertEqual(required_accumulator_bits(4, 4, 64)["minimum"], -3584)
        with self.assertRaises(ValueError):
            required_accumulator_bits(3, 4, 64)
        with self.assertRaises(ValueError):
            required_accumulator_bits(4, 4, 0)
        with self.assertRaises(ValueError):
            required_accumulator_bits(4, 4, True)

    def test_storage_requirements_include_double_buffers_and_output_width(self):
        int4 = next(mode for mode in MODES if mode.name == "int4_acc16")
        fp16_out16 = next(mode for mode in MODES if mode.name == "fp16_acc32_out16")
        bf16_out16 = next(mode for mode in MODES if mode.name == "bf16_acc32_out16")
        fp16_out32 = next(mode for mode in MODES if mode.name == "fp16_acc32_out32")

        self.assertEqual(
            storage_requirement(8, 16, int4, buffers=2, lanes=2)
            ["data_scratchpad_reserved_bytes"],
            112,
        )
        self.assertEqual(
            storage_requirement(8, 16, fp16_out16, buffers=2, lanes=2)
            ["data_scratchpad_reserved_bytes"],
            352,
        )
        self.assertEqual(
            storage_requirement(8, 16, bf16_out16, buffers=2, lanes=2)
            ["data_scratchpad_reserved_bytes"],
            352,
        )
        self.assertEqual(
            storage_requirement(8, 16, fp16_out32, buffers=2, lanes=2)
            ["data_scratchpad_reserved_bytes"],
            384,
        )
        self.assertEqual(
            storage_requirement(8, 16, int4, buffers=2, lanes=2)
            ["host_quantization_parameter_bytes"],
            40,
        )

    def test_error_metrics_expose_nonfinite_population(self):
        reference = np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
        actual = np.asarray([1.5, np.nan, np.inf, -np.inf], dtype=np.float64)
        metrics = error_metrics(reference, actual)
        self.assertEqual(metrics["elements"], 4)
        self.assertEqual(metrics["reference_nonfinite"], 3)
        self.assertEqual(metrics["nonfinite_outputs"], 3)
        self.assertFalse(metrics["complete_finite"])
        self.assertIsNone(metrics["max_abs_error"])
        self.assertEqual(metrics["finite_subset"]["max_abs_error"], 0.5)

        all_nonfinite = error_metrics(
            np.asarray([np.nan, np.inf]), np.asarray([np.nan, np.inf])
        )
        self.assertIsNone(all_nonfinite["finite_subset"])
        self.assertIsNone(all_nonfinite["rmse"])
        with self.assertRaises(ValueError):
            error_metrics(np.ones(2), np.ones(1))


@unittest.skipUnless(
    LIBRARY.is_file() and HAS_TORCH,
    "C++ precision library and torch are required for ABI probes",
)
class PrecisionAbiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = PrecisionRuntime(LIBRARY)

    @staticmethod
    def _decode_all_codes(dtype):
        import torch

        bits = np.arange(65536, dtype=np.uint16)
        signed = np.ascontiguousarray(bits.view(np.int16))
        decoded = torch.from_numpy(signed).view(getattr(torch, dtype)).to(torch.float32)
        return bits, decoded.numpy().copy()

    def _assert_conversion_matches_torch(self, dtype, values):
        with np.errstate(all="ignore"):
            expected_decoded, expected_bits = torch_convert(values, dtype)
            actual_decoded, actual_bits = self.runtime.convert(values, dtype)
        expected_nan = np.isnan(expected_decoded)
        self.assertTrue(np.all(np.isnan(actual_decoded[expected_nan])))
        non_nan = ~expected_nan
        np.testing.assert_array_equal(actual_bits[non_nan], expected_bits[non_nan])
        assert_same_float32(actual_decoded, expected_decoded, dtype + " random conversion")

    def test_all_fp16_encodings_roundtrip(self):
        bits, source = self._decode_all_codes("float16")
        decoded, encoded = self.runtime.convert(source, "float16")
        nan = np.isnan(source)
        np.testing.assert_array_equal(encoded[~nan], bits[~nan])
        self.assertTrue(np.all(np.isnan(decoded[nan])))
        assert_same_float32(decoded[~nan], source[~nan], "FP16 full encoding roundtrip")

    def test_all_bf16_encodings_roundtrip(self):
        bits, source = self._decode_all_codes("bfloat16")
        decoded, encoded = self.runtime.convert(source, "bfloat16")
        nan = np.isnan(source)
        np.testing.assert_array_equal(encoded[~nan], bits[~nan])
        self.assertTrue(np.all(np.isnan(decoded[nan])))
        assert_same_float32(decoded[~nan], source[~nan], "BF16 full encoding roundtrip")

    def test_fifty_thousand_random_float32_patterns_match_torch(self):
        rng = np.random.default_rng(0x51A7E)
        values = rng.integers(0, 2**32, size=50000, dtype=np.uint32).view(np.float32)
        self._assert_conversion_matches_torch("float16", values)
        self._assert_conversion_matches_torch("bfloat16", values)

    def test_integer_wrap_boundaries_and_error_recovery(self):
        int4_acc16 = next(mode for mode in MODES if mode.name == "int4_acc16")
        int4_acc32 = next(mode for mode in MODES if mode.name == "int4_acc32")
        int8_acc16 = NumericMode("int8_acc16", "int8", "int8", "int16", "int16")
        int8_acc32 = next(mode for mode in MODES if mode.name == "int8_acc32")

        a = np.full(512, -8.0, dtype=np.float32)
        w = a.reshape(1, -1)
        actual, counters = self.runtime.gemv(a, w, int4_acc16)
        expected, expected_counters = ordered_reference(a, w, int4_acc16)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0], -32768.0)
        self.assertEqual(counters, expected_counters)
        self.assertEqual(counters["integer_overflow_events"], 1)

        actual, counters = self.runtime.gemv(a, w, int4_acc32)
        self.assertEqual(actual[0], 32768.0)
        self.assertEqual(counters["integer_overflow_events"], 0)

        a = np.full(2, -128.0, dtype=np.float32)
        w = a.reshape(1, -1)
        actual, counters = self.runtime.gemv(a, w, int8_acc16)
        self.assertEqual(actual[0], -32768.0)
        self.assertEqual(counters["integer_overflow_events"], 1)
        actual, counters = self.runtime.gemv(a, w, int8_acc32)
        self.assertEqual(actual[0], 32768.0)
        self.assertEqual(counters["integer_overflow_events"], 0)

        bad = np.asarray([1.5, 2.0], dtype=np.float32)
        good = np.asarray([1.0, 2.0], dtype=np.float32)
        weights = np.asarray([[3.0, 4.0]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            self.runtime.gemv(bad, weights, int4_acc16)
        actual, counters = self.runtime.gemv(good, weights, int4_acc16)
        self.assertEqual(actual[0], 11.0)
        self.assertEqual(counters["integer_overflow_events"], 0)

    def test_float_gemv_order_and_nonfinite_counters(self):
        a = np.asarray([1.5, -2.25, 0.5], dtype=np.float32)
        w = np.asarray([[2.0, -1.0, 3.0], [-3.0, 4.0, 0.0]], dtype=np.float32)
        float_modes = [mode for mode in MODES if mode.activation_dtype in
                       ("float16", "bfloat16", "float32")]
        for mode in float_modes:
            actual, counters = self.runtime.gemv(a, w, mode)
            expected, expected_counters = ordered_reference(a, w, mode)
            assert_same_float32(actual, expected, mode.name + " ordered GEMV")
            self.assertEqual(counters, expected_counters)

        nonfinite_a = np.asarray([np.inf, np.nan], dtype=np.float32)
        nonfinite_w = np.asarray([[1.0, 1.0]], dtype=np.float32)
        mode = next(mode for mode in MODES if mode.name == "fp16_acc32_out32")
        actual, counters = self.runtime.gemv(nonfinite_a, nonfinite_w, mode)
        expected, expected_counters = ordered_reference(nonfinite_a, nonfinite_w, mode)
        assert_same_float32(actual, expected, "FP16 nonfinite ordered GEMV")
        self.assertEqual(counters, expected_counters)
        self.assertEqual(counters["nonfinite_product_events"], 2)
        self.assertEqual(counters["nonfinite_accumulator_events"], 2)


if __name__ == "__main__":
    unittest.main()
