# Third-Party Notices

The four simulators are vendored under `third_party/` so experiments run
without separate checkouts. Each keeps its upstream license and a `NOTICE`
that lists its source, version, and every local modification:

| Simulator | License | Files |
|---|---|---|
| EpiCare | Apache-2.0 | `third_party/epicare/LICENSE`, `third_party/epicare/NOTICE` |
| Industrial Benchmark (Python implementation) | MIT | `third_party/industrial-benchmark/LICENSE`, `third_party/industrial-benchmark/NOTICE` |
| SimBank (Zenodo v1.0.0) | CC BY 4.0 | `third_party/simbank/LICENSE`, `third_party/simbank/NOTICE` |
| NeoRL SalesPromotion (user model and transition logic) | Apache-2.0 | `third_party/sales-promotion/LICENSE`, `third_party/sales-promotion/NOTICE` |

SimBank needs two installed (not vendored) packages: `simpy` (MIT) and
`pm4py` (AGPL-3.0-or-later).

Generated artifacts (datasets, checkpoints, logs, reports) are excluded from
the source tree.
