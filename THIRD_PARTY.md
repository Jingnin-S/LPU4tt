# External projects

Original Tiny-LPU source and RTL probes use the MIT license in LICENSE.
No PDK, characterized cell library, vendor binary, or third-party source tree
is bundled.

- Tiny Tapeout template/test scaffolding and the pinned TT action/support tools:
  Apache-2.0; commits are recorded in `physical/repro/versions.lock`.  The
  files under `test/` retain their upstream template context.
- Apache TVM and Apache TVM FFI: Apache-2.0, https://github.com/apache/tvm
- PyTorch: BSD-style, https://github.com/pytorch/pytorch
- SGLang: Apache-2.0, https://github.com/sgl-project/sglang
- NumPy: BSD-3-Clause, https://numpy.org/
- pytest: MIT, https://github.com/pytest-dev/pytest
- Yosys/Icarus/Verilator and other synthesis or simulation tools retain their
  own licenses.

The optional SGLang callsite experiment retrieves a pinned source file at
runtime and records its revision/hash.  Retrieved source is not part of this
distribution.
