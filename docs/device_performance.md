# Complete-device cycles and capacity trade-offs

The server experiment executes 72 configurations: six shapes, three capacities,
one/two lanes and packed/bank4 layouts. Each architecture receives identical
inputs and weights. Independent ISA execution checks results, while independent
address/FSM phase accounting checks every C++ device tick.

RX data is continuously available and held until accepted; TX is always ready.
Counts include start, headers, checking, literals, synchronous reads, seeds,
bank conflicts, pipeline drain and writeback. External stalls and host work are
additional costs. Simulation wall time is not hardware latency.

## Two-lane bank4 results

| M x K, batch | 64B cycles | 128B cycles | 256B cycles |
| --- | ---: | ---: | ---: |
| 8 x 16, 1 | 215 | 193 | 193 |
| 8 x 16, 8 | 1152 | 907 | 907 |
| 8 x 64, 1 | 814 | 571 | 527 |
| 8 x 64, 8 | 6470 | 3370 | 2496 |
| 3 x 65, 2 | 832 | 664 | 580 |
| 3 x 256, 1 | 1454 | 1118 | 950 |

Outputs are SAT16. The existing resource/link objective selects tile and order;
these are executed candidates, not a global optimization proof for the newer
full-device objective. 128B crosses a useful short-K residency threshold.
Longer K still benefits from additional capacity and fewer partial-sum transfers.

For M3/K65/batch2 at 128B/bank4, one and two lanes both take 664 cycles.
Two-lane padding reduces tile K from 48 to 32; extra tiles, seeds and headers
cancel parallel compute gains. More lanes do not imply proportional throughput.

## Example phase ledger

128B/two lanes, M8/K64/batch8, tile rows=2, tile K=64, weight-stationary,
106B peak scratch usage and 4096 MACs:

| Phase | Cycles |
| --- | ---: |
| Start | 1 |
| Header reception | 532 |
| Boundary computation and checking | 101 |
| Independent MVM validation | 32 |
| LOAD literals | 1296 |
| STORE request/response/handshake | 192 |
| Group initialization | 32 |
| Core issue | 1024 |
| Core drain | 96 |
| Scratch writeback | 64 |
| Total | 3370 |

There is no K-tile seed cost in this case. Overall lane utilization is 60.8%
of one K element per lane per clock; a conflict-free core now issues two K
elements per lane per clock, so this figure can exceed 100%. The v0.2.1 device
took 4751 cycles here. LOAD literals and headers are now 54% of the total.
The public 64B/one-lane profile takes 7782 cycles on this workload (10887 in v0.2.1).

Use each configuration's own physical timing result when converting cycles to
time. The 100MHz qualification belongs only to the v0.2.1 64B/one-lane RTL.
Reproduce with:

~~~sh
.venv/bin/python scripts/run_device_performance.py
~~~

See [CSV](../reports/device_performance/comparison.csv),
[summary and source hashes](../reports/device_performance/summary.json),
and the [device contract](device_rtl_contract.md).
