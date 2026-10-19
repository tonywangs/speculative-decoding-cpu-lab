# GRU versus inexpensive drafts on real text

Both GRU settings were slower than their corresponding n-gram settings on every task in both seeds. The GRU training cost bought no latency advantage over this cheaper baseline in the recorded experiment.

This experiment asks whether a 23,301-parameter recurrent draft reduces greedy latency for a 289,440-parameter CPU transformer on held-out Shakespeare dialogue, Carroll prose and CPython source. It compares cached greedy, prompt lookup, training-only n-grams and GRU drafts. It is a measurement of these tiny models, not a new decoding algorithm.

Both independently trained seeds completed: 3,528 timed outputs exactly matched uncached greedy, across the same 36 held-out inputs and seven methods. These are two model pairs, three source documents, and 36 unique prompts—not independent observations per timing repeat.

## Task-by-task results

Each value is the median across cases of the median paired greedy/method latency ratio. Greater than 1 means faster; less than 1 means slower. Greedy is 1.000× by definition. No method or slowdown is omitted.

| Task | Seed | Prompt k2 | Prompt k4 | N-gram k2 | N-gram k4 | GRU k2 | GRU k4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dialogue | 17 | 1.050× | 1.070× | 1.501× | 1.526× | 1.019× | 0.947× |
| dialogue | 29 | 1.044× | 1.096× | 1.410× | 1.494× | 1.025× | 0.892× |
| prose | 17 | 1.141× | 1.151× | 1.488× | 1.715× | 0.985× | 0.971× |
| prose | 29 | 1.104× | 1.135× | 1.521× | 1.723× | 1.058× | 0.993× |
| python | 17 | 1.439× | 1.481× | 1.920× | 2.404× | 1.214× | 1.343× |
| python | 29 | 1.483× | 1.611× | 1.876× | 2.180× | 1.168× | 1.169× |

- **Dialogue** — seed 17: ngram-k4 has the highest observed median (1.526×); gru-k2 1.019×, gru-k4 0.947×; seed 29: ngram-k4 has the highest observed median (1.494×); gru-k2 1.025×, gru-k4 0.892×.
- **Prose** — seed 17: ngram-k4 has the highest observed median (1.715×); gru-k2 0.985×, gru-k4 0.971×; seed 29: ngram-k4 has the highest observed median (1.723×); gru-k2 1.058×, gru-k4 0.993×.
- **Python** — seed 17: ngram-k4 has the highest observed median (2.404×); gru-k2 1.214×, gru-k4 1.343×; seed 29: ngram-k4 has the highest observed median (2.180×); gru-k2 1.168×, gru-k4 1.169×.

These rankings are descriptive. The per-seed reports retain conditional 95% whole-case bootstrap intervals, latency, acceptance, target calls and drafting time for every method. An interval spanning 1 does not resolve speedup versus slowdown. Similar acceptance need not imply similar speed: the recurrent draft has its own inference overhead.

## Training cost and output quality

| Seed | Target training seconds | GRU training seconds | N-gram/variant fit seconds |
|---:|---:|---:|---:|
| 17 | 574.384 | 146.539 | 1.932 |
| 29 | 585.103 | 149.320 | 1.922 |

Both networks train for 1,200 updates on the same training windows; equal steps do not mean equal training compute. Costs above are the completed runs, excluding the interrupted seed-29 attempt and diagnostic work.

| Task | Seed | Target next-byte accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|---:|
| dialogue | 17 | 35.4% | 7.4% | 0.0% |
| prose | 17 | 46.2% | 9.9% | 0.0% |
| python | 17 | 64.3% | 23.6% | 0.0% |
| dialogue | 29 | 36.1% | 8.3% | 0.0% |
| prose | 29 | 46.1% | 9.6% | 0.0% |
| python | 29 | 63.4% | 21.5% | 0.0% |

Matching the target exactly is separate from predicting real continuations correctly. Repetitive, low-quality generations can be easy to draft. These results do not establish useful text generation, pretrained-LLM acceleration, semantic code completion, stochastic sampling or production performance.

## Reproduce and inspect

- [Protocol, sources, splits, seeds and numerical amendment](PROTOCOL.md).
- [Seed 17 full report](seed17/RESULTS.md) and [seed 29 full report](seed29/RESULTS.md).
- Each seed directory contains checkpoints, training traces/cost, exact source windows, compressed raw timings, correctness records, environment and hashes.
- [Installation, full reproduction, offline verification and live demo](../../README.md#real-text-gru-experiment).

Run the live demo on case 0 (dialogue), 12 (prose) or 24 (Python) to see the actual prompt, generated text, source continuation and method agreement. The recorded inputs are public and included; no provider account is needed.

Measurements use one intra-op and one inter-op CPU thread, FP32, matched target weights and kernels, two warmups and seven balanced timing repetitions. The seeds run sequentially. Shared-host load is uncontrolled; intervals condition on each target and these inputs. There is only one source per genre and the seeds share the data. Historical synthetic timings are not used as baselines for this comparison.

Regenerate this page with `.venv/bin/python scripts/summarize_realtext.py`; append `--check` to check it without writing. This verifies artifact hashes and report linkage; `verify-realtext` separately replays model correctness and recomputes summaries from raw samples.
