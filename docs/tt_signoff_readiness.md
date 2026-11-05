# Tiny Tapeout qualification gates

This release includes completed bounded SKY130A macro qualification for the
64B/one-lane configuration. See [final results](tt_signoff_results.md).
The checklist below defines the gates; it is not a new claim about another
profile, shuttle or chip.

| Stage | Required evidence |
| --- | --- |
| Toolchain | Pinned template/support tools, matching PDK and LibreLane; official example |
| Complete device RTL | Framer/checker, validity, banks/gather, MAC, writeback, status/reset |
| Functional/cycle validation | Independent C++ and arithmetic references, errors/backpressure/reset/tails |
| Configuration | Exact source/metadata/SDC/tile/clock agreement |
| Physical implementation | Synthesis, placement/CTS/routing/extraction with declared corners |
| Qualification | Setup/hold and cap/slew, DRC/LVS/antenna, precheck, gate vectors and artifact hashes |

Do not convert 45nm probe timing into SKY130 frequency claims. Do not call a
flow successful solely because GDS exists or precheck passes. Record negative
slack, electrical errors, unrun checks and resource-aborted experiments explicitly.
Use standard Tiny Tapeout ports and integration rules; do not add a duplicate
custom pad ring inside the LPU macro.

Primary pinned sources are recorded in configs/templates/sky.json and
physical/repro/versions.lock:
[SKY template](https://github.com/TinyTapeout/ttsky-verilog-template/tree/a42e6662fab316e93b4e26399e21d41664a460b7),
[GDS action](https://github.com/TinyTapeout/tt-gds-action/tree/395eedb3eeec3c2f4961c3b04ff2cf87f5525636),
[LibreLane Classic](https://github.com/librelane/librelane/blob/3.0.14/librelane/flows/classic.py).
Macro qualification does not select/pay for a shuttle or constitute submission.
