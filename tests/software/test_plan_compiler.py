import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from tinylpu_compiler import Target, compile_plan, gemv, input_vector, relu_q
from tinylpu_compiler.plan import HEADER, INSTRUCTION, pack_row, verify_artifact


class CompilerTests(unittest.TestCase):
    def test_linear_capacity_tiling_and_input_reuse(self):
        node = gemv(input_vector(16), [[(i+j)%16-8 for j in range(16)] for i in range(8)])
        with tempfile.TemporaryDirectory() as directory:
            m = compile_plan(node, directory)
            self.assertEqual([k["rows"] for k in m["kernels"]], [4,4])
            self.assertEqual(m["predicted_core_cycles"], 68)
            self.assertEqual(m["predicted_transfers"], {"load_a_bytes":8,"load_w_bytes":64,"store_y_bytes":16})
            self.assertEqual(max(k["layout"]["reserved_bytes"] for k in m["kernels"]), 64)
            data = Path(directory, "program.tlp").read_bytes()
            self.assertEqual(len(data), HEADER.size+len(m["instructions"])*INSTRUCTION.size)
            self.assertEqual(verify_artifact(directory)["abi"], 1)

    def test_mlp_has_explicit_host_step(self):
        first = gemv(input_vector(16), [[1]*16 for _ in range(4)])
        output = gemv(relu_q(first), [[1]*4 for _ in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            m = compile_plan(output, directory)
            self.assertEqual(m["host_tensor_count"], 4)
            self.assertEqual(m["predicted_host_ops"], 4)
            self.assertEqual(m["predicted_core_cycles"], 44)
            self.assertEqual(sum(i["op"] == "HOST_RELU_Q" for i in m["instructions"]), 1)

    def test_odd_k_row_packing_and_masked_tail(self):
        self.assertEqual(pack_row([-8,7,-1]), b"\x78\x0f")
        output = gemv(input_vector(17), [[-8]*17 for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            m = compile_plan(output, directory)
            self.assertEqual([k["rows"] for k in m["kernels"]], [2,1])
            self.assertEqual(m["weights_bytes"], 27)

    def test_artifact_determinism_and_tamper_rejection(self):
        output = gemv(input_vector(4), [[1,2,3,4]])
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = compile_plan(output, first), compile_plan(output, second)
            self.assertEqual(a["program_sha256"], b["program_sha256"])
            self.assertNotIn("input_values", a)
            Path(first, "weights.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "hash mismatch"): verify_artifact(first)

    def test_invalid_dtype_capacity_and_target_rejected(self):
        with self.assertRaises(ValueError): gemv(input_vector(1), [[8]])
        with self.assertRaises(ValueError): Target(lanes=3).validate()
        output = gemv(input_vector(16), [[1]*16])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No legal"): compile_plan(output, directory, Target(scratchpad_bytes=8))


if __name__ == "__main__": unittest.main()
