# Third-Party Notices

Four of the six simulators are vendored under `third_party/` so experiments
run without separate checkouts. Each keeps its upstream license and a `NOTICE`
that lists its source, version, and every local modification:

| Simulator | License | Files |
|---|---|---|
| EpiCare | Apache-2.0 | `third_party/epicare/LICENSE`, `third_party/epicare/NOTICE` |
| Industrial Benchmark (Python implementation) | MIT | `third_party/industrial-benchmark/LICENSE`, `third_party/industrial-benchmark/NOTICE` |
| SimBank (Zenodo v1.0.0) | CC BY 4.0 | `third_party/simbank/LICENSE`, `third_party/simbank/NOTICE` |
| NeoRL SalesPromotion (user model and transition logic) | Apache-2.0 | `third_party/sales-promotion/LICENSE`, `third_party/sales-promotion/NOTICE` |

SimBank needs two installed (not vendored) packages: `simpy` (MIT) and
`pm4py` (AGPL-3.0-or-later).

CityLearn and Grid2Op are installed, not vendored, from
`requirements-nodeps.txt` (their declared pins predate numpy 2 and pandas 3),
and are used unmodified:

| Package | License | Data it downloads |
|---|---|---|
| `citylearn` 3.2.0 | MIT | `citylearn_challenge_2022_phase_all`, from the CityLearn repository's v3.2.0 tag (MIT repository; the home data come from EPRI, Texas Data Repository doi:10.18738/T8/0YLJ6Q) |
| `grid2op` 1.12.6 | MPL-2.0 | `l2rpn_case14_sandbox`, from huggingface.co/datasets/grid2op/l2rpn-datasets, which states no license; the index that points to it (github.com/Grid2Op/grid2op-datasets) is MPL-2.0 |
| `lightsim2grid` 1.3.0 | MPL-2.0 | none |
| `pandapower` 3.5.6 | BSD-3-Clause | none |

Their remaining dependencies in `requirements-nodeps.txt` are MIT or BSD
(`simplejson` is MIT or AFL-2.1).

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
