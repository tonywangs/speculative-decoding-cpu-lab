# Six-unit controller cost calibration

**Calibration complete; the full two-experiment research goal remains unfinished.** No full-population experiment ran and no second-experiment branch was selected.

The six first-window validation case/seed units produced 438 durable records: 6 independent uncached references, 12 quality passes, 42 numerical audits, 84 warmups and 294 timings. All 420 audited, warmed and timed outputs matched uncached greedy exactly. Every method occupies every timing position once within each unit.

Cumulative controller execution: **85.56 s**, **29,297 forward calls** (20,401 target; 8,896 GRU), **244.37 MiB** peak resident memory. Limits were 600 execution seconds, 150,000 aggregate forwards and 4 GiB address space. No calls were lost to crashes (0 charged unknown calls).

The run paused after 10 records, then resumed the same journal in a second process. There are 2 recorded execution environments. The pause followed the first unit’s audits, so its two warmups occurred in the resumed process. All successful records and cumulative budgets survived the restart. Separate small-model tests exercise SIGTERM, a mid-timing restart with rewarming, hard memory allocation failure, atomic rollback and exact call-limit exhaustion; those are mechanism tests, not research samples.

CPU: DO-Regular; Torch 2.8.0+cpu; 1 intra-op thread, one process, FP32. The journal records both environments and source hashes. The host is shared.

## Full workload requirement

The manifest preserves 831 validation and 832 test windows, both seeds, the original training windows and source/model identities. One Python validation window is excluded because its prompt occurs in its source training region. Calibration does not remove any full-study work.

| Work over both full experiments | Count | Extrapolated compute hours |
|---|---:|---:|
| Timed decodes | 325,948 | 11.143 |
| Warmup decodes | 93,128 | 3.056 |
| Audited decodes | 46,564 | 5.277 |
| Shared uncached references | 3,326 | 0.374 |
| Shared target/GRU quality passes | 6,652 | 0.016 |

The source/seed-weighted cost analogue is **25.77 hours** and **32,233,742 forward calls**, including 5.90 extrapolated hours of setup, durable checkpointing and orchestration. This exceeds the prior two-hour/two-million-call caps by **23.77 hours** and **30,233,742 calls** before any additional safety margin.

Median measured result-commit cost was 13.04 ms. The 2 recorded derived-view exports took 1.105 s in total outside the execution budget. SQLite commits, rather than JSON views, are the resumable checkpoints. The overhead analogue spreads observed loading/restart and per-operation costs across the full workload; it is deliberately not a precise optimized-runtime forecast.

Both predeclared second-experiment branches retain seven configurations and the same full population. Each is assigned this measured experiment-1 cost analogue, not a measured branch runtime. The current decoding-path worst-case forward-count bound for either complete route is 66,992,292 (64-token outputs; maximum four draft tokens per block). Neither branch has a proven wall-time bound. The observed memory peak fits the cap for serial calibration, not an unmeasured parallel run.

The resource shortfall is actionable: a future full evaluation needs a separately authorized larger cumulative budget with headroom, or a measured implementation improvement that preserves the population and protocol. This run stops here.

## Observed timings on the six calibration units

Each cell is the median of seven paired cached-greedy/method latency ratios. Values below one are slowdowns. There is only **one window per source and seed**; these are operational observations, not population speed estimates or confidence intervals.

| Method | 17 dialogue | 17 prose | 17 Python | 29 dialogue | 29 prose | 29 Python |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000× | 1.000× |
| prompt-k2 | 1.138× | 0.995× | 1.251× | 1.158× | 1.062× | 1.278× |
| prompt-adaptive | 1.101× | 1.003× | 1.247× | 1.082× | 0.991× | 1.310× |
| ngram-k2 | 1.529× | 1.476× | 1.427× | 1.500× | 1.573× | 1.618× |
| ngram-adaptive | 1.357× | 1.535× | 1.510× | 1.446× | 1.572× | 1.578× |
| gru-k2 | 0.990× | 0.882× | 0.965× | 0.983× | 0.978× | 1.061× |
| gru-adaptive | 0.993× | 0.892× | 0.927× | 0.992× | 1.026× | 1.088× |

Cached-greedy median request latency (same column order): 136.50 ms, 135.81 ms, 141.76 ms, 136.96 ms, 134.91 ms, 134.46 ms.

N-gram variants were faster than cached greedy in these six units. Adaptive lengths did not consistently improve on fixed k2. GRU variants showed both speedups and slowdowns. These six cases do not authorize selecting the second experiment: that rule requires complete validation across all 831 validation windows and both seeds.

## Quality is distinct from output agreement

| Seed / source | Target next-byte accuracy | GRU next-byte accuracy | Target bits/byte | GRU bits/byte | Free-running byte accuracy |
|---|---:|---:|---:|---:|---:|
| 17 / dialogue | 31.2% | 32.8% | 3.541 | 3.033 | 4.7% |
| 17 / prose | 68.8% | 59.4% | 1.622 | 2.270 | 9.4% |
| 17 / python | 57.8% | 67.2% | 2.175 | 2.334 | 15.6% |
| 29 / dialogue | 28.1% | 35.9% | 3.528 | 2.919 | 9.4% |
| 29 / prose | 57.8% | 65.6% | 1.850 | 2.077 | 14.1% |
| 29 / python | 62.5% | 51.6% | 2.017 | 2.154 | 14.1% |

No generated continuation exactly matched its 64-byte source continuation. These tiny models do not produce reliably useful language. A faster reproduction of a wrong or repetitive target output does not improve generation quality.

## A visible saved example

Seed 17, `validation-python-0051160`. All seven configurations produced the same target continuation.

Public CPython prompt:

```text
or_precedence is not _Precedence.FACTOR:
                self.write(" ")
            self.set_pr
```

Generated target continuation:

```text
ecedence.__delit___dencedence


  def visit_MESSAGELUS:
```

Actual source continuation:

```text
ecedence(operator_precedence, node.operand)
            self.tra
```

## Inspect or reproduce

```bash
.venv/bin/python -m specdecode.research status
.venv/bin/python -m specdecode.research verify
.venv/bin/python scripts/check_controller.py
.venv/bin/python scripts/summarize_controller.py --check
```

See the [controller guide](../../docs/research-controller.md) for initialization, deadline/resume commands, resource semantics and the predeclared branches. [Raw measurements](measurements.jsonl), [extrapolation](extrapolation.json), [population](population.json), [proposal policy](proposal-policy.json) and [state](state.json) accompany the authoritative SQLite journal. Exact agreement and numerical-audit diagnostics apply to these outputs only; no pretrained-LLM, unseen-document, production-serving or sampling claim is made.
