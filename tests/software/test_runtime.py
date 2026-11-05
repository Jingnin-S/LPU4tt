import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"compiler"))
from tinylpu_compiler import compile_plan, gemv, input_vector, relu_q
from tinylpu_compiler.runtime import PlanSession

LIBRARY = Path(os.environ.get("TINYLPU_RUNTIME_LIBRARY", ROOT/"build_esl/libtinylpu_plan_runtime.so"))
AVAILABLE = LIBRARY.is_file() and importlib.util.find_spec("numpy") is not None
VM_AVAILABLE = AVAILABLE and importlib.util.find_spec("tvm") is not None


@unittest.skipUnless(AVAILABLE, "Built Tiny-LPU shared library and numpy are required")
class SessionTests(unittest.TestCase):
    def test_reuse_validation_recovery_and_close(self):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as folder:
            manifest = compile_plan(gemv(input_vector(4), [[1]*4, [-1]*4]), folder)
            with PlanSession(folder, LIBRARY) as session:
                self.assertEqual(session.run(np.zeros((1,4), dtype="float32"))["result"], [0,0])
                self.assertEqual(session.run(np.full((1,4),7,dtype="float32"))["result"], [28,-28])
                for bad in (np.full((1,4), 8, dtype="float32"), np.ones((1,4), dtype="float64"),
                            np.full((1,4), .5, dtype="float32"), np.ones(4, dtype="float32")):
                    with self.assertRaises(ValueError):
                        session.run(bad)
                self.assertEqual(session.run(np.zeros((1,4), dtype="float32"))["result"], [0,0])
                self.assertEqual(session.manifest["program_sha256"], manifest["program_sha256"])
                with ThreadPoolExecutor(max_workers=1) as executor:
                    with self.assertRaisesRegex(RuntimeError, "creating thread"):
                        executor.submit(session.close).result()
                self.assertEqual(session.run(np.zeros((1,4), dtype="float32"))["result"], [0,0])
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.run(np.zeros((1,4), dtype="float32"))

    def test_corrupt_artifact_fails_before_load(self):
        with tempfile.TemporaryDirectory() as folder:
            compile_plan(gemv(input_vector(4), [[1]*4]), folder)
            path = Path(folder)/"weights.bin"
            path.write_bytes(b"\x00"*len(path.read_bytes()))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                PlanSession(folder, LIBRARY)


@unittest.skipUnless(VM_AVAILABLE, "Compiler venv and built Tiny-LPU shared library are required")
class VmTests(unittest.TestCase):
    def test_vm_linear_uses_device_and_repeats(self):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from tinylpu_compiler.vm import VmProgram

        with tempfile.TemporaryDirectory() as folder:
            node = gemv(input_vector(4), [[1]*4, [-1]*4])
            with VmProgram(node, folder, LIBRARY) as program:
                for value in (-8, 0, 7, 1):
                    result = program.run(np.full((1,4), value, dtype="float32"))
                    self.assertEqual(result.tolist(), [[value*4, -value*4]])
                    self.assertEqual(len(program.last_reports), 1)
                    self.assertEqual(program.last_reports[0]["core_cycles"], 6)
                self.assertEqual(program.total_callback_calls, 4)
                with self.assertRaises(ValueError):
                    program.run(np.full((1,4), 8, dtype="float32"))
                self.assertEqual(program.last_reports, [])
                self.assertEqual(program.last_timing, {})
                with ThreadPoolExecutor(max_workers=1) as executor:
                    with self.assertRaisesRegex(RuntimeError, "creating thread"):
                        executor.submit(program.close).result()
                self.assertEqual(program.run(np.zeros((1,4), dtype="float32")).tolist(), [[0,0]])

    def test_mlp_host_quantization_is_native_vm(self):
        import numpy as np
        from tinylpu_compiler.vm import VmProgram

        first = gemv(input_vector(16), [[1]*16, [-1]*16, [0]*16, [0]*16])
        output = gemv(relu_q(first), [[1]*4])
        with tempfile.TemporaryDirectory() as folder:
            with VmProgram(output, folder, LIBRARY) as program:
                self.assertEqual(program.run(np.ones((1,16), dtype="float32")).tolist(), [[7]])
                self.assertEqual(len(program.last_reports), 2)
                self.assertEqual(sum(report["host_ops"] for report in program.last_reports), 0)
                self.assertEqual(program.host_quant_elements, 4)
                ir = Path(folder, "vm_relax.ir.txt").read_text()
                self.assertIn("floor", ir)
                self.assertIn("clip", ir)
                self.assertIn("call_dps_packed", ir)


if __name__ == "__main__":
    unittest.main()
