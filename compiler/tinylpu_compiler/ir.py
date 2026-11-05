from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(eq=False)
class TensorNode:
    op: str
    size: int
    bits: int
    source: Optional["TensorNode"] = None
    weights: Tuple[Tuple[int, ...], ...] = ()


def input_vector(size):
    if type(size) is not int or not 1 <= size <= 4096:
        raise ValueError("Invalid input-vector size")
    return TensorNode("input", size, 4)


def gemv(source, weights):
    rows = tuple(tuple(row) for row in weights)
    if source.bits != 4 or not rows or any(len(row) != source.size for row in rows):
        raise ValueError("GEMV requires an INT4 source and matching row-major weights")
    if any(type(x) is not int or not -8 <= x <= 7 for row in rows for x in row):
        raise ValueError("Weights must be signed INT4 integers")
    if not 1 <= len(rows) <= 4096:
        raise ValueError("Invalid output-vector size")
    return TensorNode("gemv", len(rows), 16, source, rows)


def relu_q(source):
    if source.bits != 16:
        raise ValueError("Explicit ReLU/3-4 requant expects an INT16 source")
    return TensorNode("host_relu_q34", source.size, 4, source)
