# ISA experiment results

Actual encoded streams; independent integer oracle and exact cycle/byte checks.
Timing is a sequential scenario estimate, not measured hardware.

```json
{
  "fixtures": [
    {
      "name": "linear_1x1",
      "rows": 1,
      "k": 1,
      "runs": 96,
      "source_sha256": "b10f237e415e8dcb939d90577c8d7be99ace37498849f9dd1fe9d2b7a748a4cb"
    },
    {
      "name": "linear_17x3",
      "rows": 3,
      "k": 17,
      "runs": 192,
      "source_sha256": "bc2889a3424683d19cfd2d398a92301ff2957d5b7c281498857f50de0fdd4dcd"
    },
    {
      "name": "linear_16x7",
      "rows": 7,
      "k": 16,
      "runs": 96,
      "source_sha256": "0c5e69776ced7ab96d73e6b043b1745b788490965929b9812376fa91a7ecf8ea"
    },
    {
      "name": "linear_16x8",
      "rows": 8,
      "k": 16,
      "runs": 192,
      "source_sha256": "8fa5c4efb445bc320ca2ab94bec47971ee4ca573a946a73bd13c3add6109da31"
    },
    {
      "name": "linear_64x8",
      "rows": 8,
      "k": 64,
      "runs": 192,
      "source_sha256": "473caadbd3983695cc04bcc99e68a92097b1e1ca0f1122cca5ea3a1356bd1eb9"
    },
    {
      "name": "linear_65x8",
      "rows": 8,
      "k": 65,
      "runs": 96,
      "source_sha256": "7ec7ce91cd2bbf93e13e322bc6cd063ed7152bf50b7dd0a5546746916be27797"
    },
    {
      "name": "linear_256x8",
      "rows": 8,
      "k": 256,
      "runs": 192,
      "source_sha256": "6134365cda06acd6129af3b84992912075c2658954c27d70b7d5ef129c905300"
    },
    {
      "name": "linear_1024x4",
      "rows": 4,
      "k": 1024,
      "runs": 96,
      "source_sha256": "0b7a652e6f7f8e99d05937f3b019266f2c903ff64e15ec46fba3b1ab6e351bc5"
    },
    {
      "name": "linear_4096x3",
      "rows": 3,
      "k": 4096,
      "runs": 192,
      "source_sha256": "6131618a5fd6fc1bb65e4256de23e8c425e37b1b0a67e7a0b9a909fc5af39e71"
    }
  ],
  "comparison_rows": 1344,
  "tiling_candidates": 13108,
  "factor_rows": 336,
  "legacy_checks": 10,
  "diagnostic_runs": 31,
  "mlp_graphs": 2,
  "cpp_runs": 1379,
  "errors": 0,
  "physical_performance_validated": false
}
```

Primary input values are identical across INT4/INT8 to isolate storage/transport; full-range INT8 is in diagnostics.
Tile selection minimizes modeled time at 100MHz / 10MB/s / zero host dispatch, within the listed candidate set.
Both encodings use the geometry chosen for mixed32/64; no claim of globally optimal tiling.
Internal scratch copy, group drains and partial-sum traffic are counted; no asynchronous overlap.
All frames arrive at t=0; aggregation wait and host quantization time are excluded.
