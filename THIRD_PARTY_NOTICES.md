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

The `tabpfn` representation needs the installed (not vendored) `tabpfn`
package (Apache-2.0). The pretrained weights it downloads are licensed
separately: TabPFN-3.5, the version in `configs/protocol.json`, is under the
non-commercial TabPFN-3.5 License v1.0
(https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE), which each
user accepts at https://ux.priorlabs.ai before the first download. Without
that acceptance the representation falls back to the TabPFN-2 weights, which
download without it.

Generated artifacts (datasets, checkpoints, logs, reports) are excluded from
the source tree.
