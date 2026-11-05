"""Actual torch.export -> TVM Relax BYOC partition -> Tiny-LPU IR import.

Only the explicitly declared float32 carrier of exact INT4 integers is accepted.
This prototype emits its own host plan, not a TVM VM external runtime module.
"""

from dataclasses import dataclass
from pathlib import Path

from .ir import TensorNode, gemv, input_vector, relu_q


@dataclass(eq=False)
class Expr:
    op: str
    args: tuple
    attrs: object = None


@dataclass
class Closure:
    function: object
    scope: dict


def import_model(model, example_args, output_dir, multi_output=False, allow_batch=False):
    import numpy as np
    import torch
    import tvm
    from tvm import relax
    from tvm.relax.dpl.pattern import is_op, wildcard
    from tvm.relax.frontend.torch import from_exported_program

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    exported = torch.export.export(model.eval(), example_args)
    module = from_exported_program(exported, keep_params_as_input=False, unwrap_unit_return_tuple=True)
    (folder / "imported_relax.ir.txt").write_text(module.script(), encoding="utf-8")
    pattern = is_op("relax.matmul")(wildcard(), wildcard())
    partitioned = relax.transform.FuseOpsByPattern(
        [("tinylpu.matmul", pattern)], bind_constants=False, annotate_codegen=True)(module)
    partitioned = relax.transform.MergeCompositeFunctions()(partitioned)
    (folder / "partitioned_relax.ir.txt").write_text(partitioned.script(), encoding="utf-8")
    external = []
    for symbol, function in partitioned.functions.items():
        attrs = getattr(function, "attrs", None)
        if attrs is not None and "Codegen" in attrs and str(attrs["Codegen"]) == "tinylpu":
            external.append(symbol.name_hint)
    if not external:
        raise ValueError("No actual Tiny-LPU BYOC partition was produced")

    main = partitioned["main"]
    if len(main.params) != 1:
        raise ValueError("ABI v1 frontend requires one runtime tensor; parameters must be bound")
    info = getattr(main.params[0], "struct_info", None)
    if info is None:
        info = main.params[0].ty
    shape = info.shape
    dims = [int(v) for v in (shape.values if hasattr(shape, "values") else shape)]
    if len(dims) != 2 or (dims[0] != 1 and not allow_batch) or str(info.dtype) != "float32":
        raise ValueError("Expected a [1,K] float32 carrier of exact INT4 values")
    input_node = input_vector(dims[1])

    def var_name(var):
        return getattr(var, "name_hint", None) or var.name

    def const_scalar(value):
        if isinstance(value, np.ndarray) and value.size == 1:
            return float(value.reshape(-1)[0])
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError("Expected scalar constant")

    def evaluate(expr, scope):
        if isinstance(expr, relax.Var):
            return scope[var_name(expr)]
        if isinstance(expr, getattr(relax, "Constant", ())):
            return expr.data.numpy()
        if isinstance(expr, getattr(tvm.ir, "GenericConst", ())):
            value = expr.value
            return value.numpy() if hasattr(value, "numpy") else value
        if isinstance(expr, getattr(relax, "PrimValue", ())):
            return float(expr.value.value)
        if type(expr).__name__ in ("IntImm", "FloatImm"):
            return float(expr.value)
        if isinstance(expr, relax.Function):
            return Closure(expr, dict(scope))
        if isinstance(expr, relax.Tuple):
            values = [evaluate(v, scope) for v in expr.fields]
            if len(values) != 1:
                if multi_output:
                    return tuple(values)
                raise ValueError("Only one graph output is supported")
            return values[0]
        if isinstance(expr, relax.TupleGetItem):
            if multi_output:
                value = evaluate(expr.tuple_value, scope)
                if isinstance(value, tuple):
                    return value[int(expr.index)]
            if expr.index != 0:
                raise ValueError("Unsupported tuple projection")
            return evaluate(expr.tuple_value, scope)
        if isinstance(expr, relax.SeqExpr):
            local = dict(scope)
            for block in expr.blocks:
                for binding in block.bindings:
                    if not isinstance(binding, relax.VarBinding):
                        raise ValueError("Unsupported Relax binding")
                    local[var_name(binding.var)] = evaluate(binding.value, local)
            return evaluate(expr.body, local)
        if isinstance(expr, relax.Call):
            arguments = [evaluate(v, scope) for v in expr.args]
            if isinstance(expr.op, tvm.ir.GlobalVar):
                function = partitioned[expr.op]
                local = {var_name(p):a for p,a in zip(function.params, arguments)}
                return evaluate(function.body, local)
            if isinstance(expr.op, relax.Function):
                local = dict(scope)
                local.update({var_name(p):a for p,a in zip(expr.op.params, arguments)})
                return evaluate(expr.op.body, local)
            if isinstance(expr.op, relax.Var):
                closure = evaluate(expr.op, scope)
                if not isinstance(closure, Closure):
                    raise ValueError("Indirect Relax call is not a composite function")
                local = dict(closure.scope)
                local.update({var_name(p):a for p,a in zip(closure.function.params, arguments)})
                return evaluate(closure.function.body, local)
            name = str(expr.op.name)
            if name == "relax.permute_dims" and isinstance(arguments[0], np.ndarray):
                axes = getattr(expr.attrs, "axes", None)
                return np.transpose(arguments[0], axes=None if axes is None else tuple(int(v) for v in axes))
            if name == "relax.reshape" and isinstance(arguments[0], np.ndarray):
                return np.reshape(arguments[0], tuple(int(v) for v in expr.args[1].values))
            if name in ("relax.matmul", "relax.nn.relu", "relax.multiply", "relax.add", "relax.floor", "relax.clip"):
                return Expr(name, tuple(arguments), expr.attrs)
            raise ValueError("Unsupported Relax operation: " + name)
        if isinstance(expr, relax.ShapeExpr):
            return tuple(int(v) for v in expr.values)
        raise ValueError("Unsupported Relax expression " + type(expr).__name__)

    expression = evaluate(main.body, {var_name(main.params[0]): input_node})
    memo = {}

    def scalar_equals(value, expected):
        try:
            return const_scalar(value) == expected
        except ValueError:
            return False

    def peel_binary(node, operation, scalar):
        if not isinstance(node, Expr) or node.op != operation:
            raise ValueError("Explicit requantization pattern mismatch")
        if scalar_equals(node.args[1], scalar): return node.args[0]
        if scalar_equals(node.args[0], scalar): return node.args[1]
        raise ValueError("Requantization requires the declared exact scalar")

    def lower(expr):
        if multi_output and isinstance(expr, tuple):
            return tuple(lower(v) for v in expr)
        if isinstance(expr, TensorNode): return expr
        if not isinstance(expr, Expr): raise ValueError("Graph value is not supported tensor IR")
        if expr in memo: return memo[expr]
        if expr.op == "relax.matmul":
            source = lower(expr.args[0])
            weight = expr.args[1]
            if not isinstance(weight, np.ndarray) or weight.ndim != 2 or weight.shape[0] != source.size:
                raise ValueError("Matmul weight must be a static [K,N] matrix")
            if not np.all(np.isfinite(weight)) or not np.all(weight == np.round(weight)) or np.min(weight) < -8 or np.max(weight) > 7:
                raise ValueError("Weight is outside the exact INT4 subset")
            result = gemv(source, [[int(v) for v in row] for row in weight.T])
        elif expr.op == "relax.clip":
            clip_min = const_scalar(expr.args[1]) if len(expr.args) == 3 else float(expr.attrs.a_min)
            clip_max = const_scalar(expr.args[2]) if len(expr.args) == 3 else float(expr.attrs.a_max)
            if clip_min != 0 or clip_max != 7:
                raise ValueError("Only explicit ReLU/3-4 INT4 requant is supported")
            floored = expr.args[0]
            if not isinstance(floored, Expr) or floored.op != "relax.floor":
                raise ValueError("Missing explicit quantization rounding")
            multiplied = peel_binary(floored.args[0], "relax.add", 0.5)
            activated = peel_binary(multiplied, "relax.multiply", 0.75)
            if not isinstance(activated, Expr) or activated.op != "relax.nn.relu":
                raise ValueError("Missing ReLU in declared requantization pattern")
            result = relu_q(lower(activated.args[0]))
        else:
            raise ValueError("Unsupported graph output/fusion: " + expr.op)
        memo[expr] = result
        return result

    output = lower(expression)
    metadata = {"torch_version": torch.__version__, "tvm_version": tvm.__version__,
                "exported_input_shape": dims, "byoc_functions": external,
                "frontend": "torch.export -> Relax importer -> FuseOpsByPattern -> MergeCompositeFunctions",
                "codegen_input": "actual partitioned Relax module",
                "runtime_boundary": "independent Tiny-LPU command plan; TVM VM external module not linked",
                "numerical_contract": "float32 carrier, runtime signed INT4 integers, constant INT4 weights"}
    return output, metadata
