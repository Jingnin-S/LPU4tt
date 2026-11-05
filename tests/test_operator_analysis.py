import importlib.util
from pathlib import Path
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("analyze_operators", ROOT / "scripts/analyze_operators.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


class OperatorOracleTests(unittest.TestCase):
    def test_matrix_and_transpose_layout(self):
        self.assertEqual(ops.integer_oracle("gemm", [2,2,2], [1,2,3,4], [5,6,7,-8]), [19,-10,43,-14])
        self.assertEqual(ops.integer_oracle("transpose", [2,3], [1,2,3,4,5,6], []), [1,4,2,5,3,6])
        self.assertEqual(ops.integer_output_shape("transpose", [2,3]), [3,2])

    def test_fir_reversal_and_requant_ties(self):
        self.assertEqual(ops.integer_oracle("fir", [4,2], [1,2,3,4], [1,2]), [4,7,10])
        self.assertEqual(ops.integer_oracle("requant", [7], [-6,-2,0,2,6,100,-100], []), [-5,-2,0,2,5,7,-8])

    def test_convolution_layouts(self):
        a = [-4,-3,-2,-1,0,1,2,3,4]
        self.assertEqual(ops.integer_oracle("conv2d", [3,3,1,1,2,2], a, [1,0,0,-1]), [-4]*4)
        # Two depthwise channels remain separate; no cross-channel sum.
        x = [1,2,3,4,5,6,7,-8]
        self.assertEqual(ops.integer_oracle("dwconv2d", [2,2,2,2,2], x, [1,-1]*4), [16,-4])

    def test_pool_valid_stride_and_negative_reductions(self):
        self.assertEqual(ops.integer_oracle("maxpool2d", [3,3,1,2,2,2,2], [-4,-3,-2,-1,0,1,2,3,4], []), [0])
        self.assertEqual(ops.integer_output_shape("maxpool2d", [3,3,1,2,2,2,2]), [1,1,1])
        self.assertEqual(ops.integer_oracle("reduce_max", [3], [-8,-7,-3], []), [-3])
        self.assertEqual(ops.integer_oracle("reduce_sum", [3], [-8,-7,-3], []), [-18])

    def test_softmax_stability_and_normalization(self):
        a = ops.nonlinear_oracle("softmax", [1,3], [10000,10001,10002], [1]*3, [0]*3, 1e-5)
        b = ops.nonlinear_oracle("softmax", [1,3], [0,1,2], [1]*3, [0]*3, 1e-5)
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(a), 1)

    def test_normalization_affine_and_epsilon(self):
        y = ops.nonlinear_oracle("layernorm", [1,3], [2,2,2], [2,3,4], [1,2,3], 1e-5)
        self.assertEqual(y, [1,2,3])
        z = ops.nonlinear_oracle("rmsnorm", [1,2], [0,0], [1,2], [0,0], 1e-5)
        self.assertEqual(z, [0,0])
        s = ops.nonlinear_oracle("sigmoid", [1,3], [-10000,0,10000], [1]*3, [0]*3, 1e-5)
        self.assertEqual(s, [0,0.5,1])

    def test_rejects_incorrect_integer_results_and_port_claims(self):
        case = {"op":"dot", "shape":[2]}
        resource = {"lanes":2, "depth":3, "a_ports":1, "b_ports":2}
        sample = dict(resource, op="dot", shape=[2], output_shape=[1], a=[1,-2], b=[3,4], golden=[-5],
                      cycle_result=[-5], peak_a_reads=1, peak_b_reads=2, model_kind="integer_bit_exact",
                      macs=2, scalar_ops=0, input_bytes=1, weight_bytes=1, output_bytes=2)
        self.assertEqual(ops.validate_integer(sample, case, resource), [-5])
        for change in ({"cycle_result":[5]}, {"peak_a_reads":2}, {"output_shape":[2]}):
            with self.assertRaises(ValueError): ops.validate_integer(dict(sample, **change), case, resource)

    def test_nonfinite_and_false_hardware_precision_fail(self):
        sample = dict(op="tanh", shape=[1,1], x=[0], gamma=[1], beta=[0], epsilon=1e-5,
                      golden=[0], cycle_result=[0], hardware_precision=None)
        self.assertEqual(ops.validate_nonlinear(sample), [0])
        for change in ({"cycle_result":[math.nan]}, {"hardware_precision":64}):
            with self.assertRaises(ValueError): ops.validate_nonlinear(dict(sample, **change))


if __name__ == "__main__": unittest.main()
