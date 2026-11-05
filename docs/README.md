# Documentation

All documents and published figures in this release are in English.

## Current specifications

- [ISA v0.1](isa_v0.md): byte encoding, arithmetic, addressing and errors.
- [Numeric contract](isa_numeric_contract.md): base, legacy and reference types.
- [Modeling guide](modeling.md): model layers, assumptions and reproduction.
- [Device contract](device_rtl_contract.md): GPIO, reset, FSM and exact cycles.
- [Device performance](device_performance.md): measured model cycles and trade-offs.
- [Architecture overview](../architecture.md): current system boundary and profiles.
- [Compiler integration](../compiler.md): Relax, composition and SGLang debugging.
- [Operator models](operator_model_spec.md): integer operators and nonlinear DAGs.
- [GPIO and clock notes](gpio_clock_notes.md): internal STA versus external I/O.

## Experiments

- [ISA experiment contract](isa_experiment_spec.md) and [decisions](isa_results.md).
- [Precision experiment contract](precision_experiment_spec.md) and [results](precision_results.md).
- [Resource experiment contract](architecture_experiment_spec.md).
- [Runtime experiment contract](runtime_experiment_spec.md).
- [Validation](../validation.md) and [reproduction](../REPRODUCIBILITY.md).
- [Compact report index](../reports/README.md).

## Physical evidence and historical design studies

- [Qualified macro results](tt_signoff_results.md).
- [Qualification gates](tt_signoff_readiness.md).
- [Physical development summary](physical_progress.md).
- [Physical environment](../physical/repro/README.md).
- [Historical architecture study](architecture_v1.md).
- [Historical architecture validation](architecture_validation.md).

## Release and figure

- [v0.2.1 release notes](release_notes_v0_2_1.md).
- [Architecture PNG](diagrams/lpu4tt_architecture.png),
  [SVG](diagrams/lpu4tt_architecture.svg),
  [editable Excalidraw](diagrams/lpu4tt_architecture.excalidraw).
