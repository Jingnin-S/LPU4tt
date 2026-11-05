from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tvm") is not None


@unittest.skipUnless(AVAILABLE, "Compiler venv with CPU PyTorch and TVM is required")
class FrontendTests(unittest.TestCase):
    def test_real_relax_partition_and_bad_weights(self):
        import torch
        from tinylpu_compiler.frontend import import_model

        class Model(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("w", torch.full((4,16),value,dtype=torch.float32))
            def forward(self,x): return torch.nn.functional.linear(x,self.w)

        with tempfile.TemporaryDirectory() as directory:
            node, metadata = import_model(Model(1), (torch.zeros(1,16),), directory)
            self.assertEqual(node.op, "gemv")
            self.assertEqual(node.size, 4)
            self.assertTrue(metadata["byoc_functions"])
            self.assertIn("tinylpu", Path(directory,"partitioned_relax.ir.txt").read_text())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"INT4"):
                import_model(Model(0.5), (torch.zeros(1,16),), directory)

    def test_unsupported_batch_and_unquantized_relu_fail(self):
        import torch
        from tinylpu_compiler.frontend import import_model

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w",torch.ones(4,16))
            def forward(self,x): return torch.relu(torch.nn.functional.linear(x,self.w))

        for shape in ((2,16),(1,16)):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    import_model(Model(), (torch.zeros(*shape),), directory)


if __name__ == "__main__": unittest.main()
