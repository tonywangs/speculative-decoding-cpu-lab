# Speculative decoding CPU lab

An installable CPU experiment comparing exact greedy decoding with prompt lookup,
a learned n-gram draft, and a small GRU draft. The new [real-text comparison](results/realtext-gru/RESULTS.md)
uses held-out Shakespeare dialogue, Carroll prose and CPython source. All model
training and inference run locally; the pinned public inputs are included.

## Real-text GRU experiment

Install with the instructions below, then show a live comparison on a held-out
excerpt using the recorded models:

```bash
.venv/bin/specdecode demo-realtext results/realtext-gru/seed17 --corpus results/realtext-gru/corpus --case-index 0
```

The demo prints the prompt, the tiny target's generated continuation, the actual
source continuation, and agreement across all seven methods. Try case indices
0–11 for dialogue, 12–23 for prose, and 24–35 for Python. Output quality is limited;
matching the target exactly does not mean predicting the source correctly.

Reproduce both paired model seeds (use new output directories):

```bash
.venv/bin/specdecode realtext --corpus results/realtext-gru/corpus --output .runs/realtext-seed17 --seed 17
.venv/bin/specdecode realtext --corpus results/realtext-gru/corpus --output .runs/realtext-seed29 --seed 29
```

Each run trains a width-96 transformer and width-32 GRU for 1,200 steps on the same
training split, then audits and times cached greedy, prompt lookup, n-gram and GRU
drafts (lengths 2/4). The [protocol](results/realtext-gru/PROTOCOL.md) specifies
source licenses/hashes, contiguous train/validation/test splits, seeds, models,
timing order, statistics and limits. The two runs share inputs but have separately
trained weights. Every timed output is checked against uncached greedy. No network
is used after package installation.

Replay saved source/split integrity, checkpoints, exact agreement, timing pairing
and report calculations without retraining or claiming to reproduce old timings:

```bash
.venv/bin/specdecode verify-realtext results/realtext-gru/seed17 --corpus results/realtext-gru/corpus
.venv/bin/specdecode verify-realtext results/realtext-gru/seed29 --corpus results/realtext-gru/corpus
```

Check the installed real-text workflow from outside the checkout:

```bash
.venv/bin/python scripts/verify_realtext_wheel.py
.venv/bin/python scripts/summarize_realtext.py --check
```

The wheel check uses the installed Torch runtime, builds and installs this package
in a temporary directory, then runs the GRU tests, replays both recorded experiments
and shows live examples from all three genres. It needs public build dependencies;
it does not reinstall Torch. The summary check verifies that the comparison page
matches the saved reports. The full unit suite below also covers shared decoder
behavior, including rejecting a changed winning token despite close logits.

Raw measurements and correctness are deterministic gzip JSON/JSONL; decompress
with Python's `gzip` module. Each run also retains both checkpoints, exact token
windows with source offsets, config, training trace/cost, validation/test quality,
hardware and source hashes, a readable report, and an integrity manifest. A failed
run has no complete manifest and is not a completed experiment.

The older [synthetic dataset replication](results/cpu-dataset-replication/RESULTS.md),
[six-target comparison](results/cpu-multiseed/RESULTS.md), and
[original experiment](results/cpu-seed17/RESULTS.md) remain historical evidence.
Their timings are not baselines for the real-text experiment. The sections below
document those original commands and synthetic designs.

## Run it

Python 3.10+ with `venv` and `pip` is required. The recorded experiment uses Python 3.12 and PyTorch 2.8.0 CPU. Install the CPU wheel first to avoid downloading CUDA dependencies on Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install .
.venv/bin/specdecode run --output .runs/example
```

That last command generates both data splits, trains the target, fits the n-gram model on training data, audits decoding, warms all methods, and writes repeated paired measurements. Use a new/empty output directory for each run. No network is used by the experiment. Dependency installation needs access to public package indexes; Debian installations may need their Python `venv` package enabled first. PyTorch can emit a harmless optional-NumPy warning if NumPy is absent; this project does not use NumPy.

For the exact tested Python 3.12 / Linux x86_64 dependency versions, replace the torch installation command with `.venv/bin/python -m pip install -r requirements-cpu.txt`. This lock uses the public CPU wheel index and is also exercised by the isolated installation check.

For a quick smoke run (not a useful target-quality estimate):

```bash
.venv/bin/specdecode run --output .runs/smoke --steps 8 --train-per-task 12 --eval-per-task 2 --warmups 1 --repeats 2
```

Inspect `RESULTS.md` and `report.json` in the output. Replay correctness without training, or repeat timings with the same model and data:

```bash
.venv/bin/specdecode verify .runs/example
.venv/bin/specdecode benchmark .runs/example --output .runs/retimed
```

## Replicate across independently generated datasets

The [dataset replication](results/cpu-dataset-replication/RESULTS.md) uses one fixed
width-64 target, three independently generated datasets, and three paired training
seeds: nine trained targets. The [frozen protocol](results/PROTOCOL.md) records the
development pilot, existing work, selection rules and analysis limits. Historical
single-seed and six-target results remain available below.

Across the nine targets, length-8 prompt lookup achieved 5.140–5.373× on repetition.
All speculative settings had median slowdowns on unpredictable data (0.863–0.947×);
one conditional interval included parity. See the [results overview](results/README.md)
for target accuracy, dataset variation and uncertainty. These synthetic CPU findings
do not establish natural-language or large-model performance.

After installing the package, reproduce the complete final experiment with:

```bash
.venv/bin/specdecode replicate --protocol results/cpu-dataset-replication/protocol.json --output .runs/dataset-replication
```

The protocol freezes 800 training steps per target, dataset seeds 3101/6203/9307,
training seeds 17/29/43, 128 held-out examples per task, the first 16 IDs per task
for timing, draft lengths 2/8, two warmups and six paired repetitions. Each length
uses its own greedy/prompt/n-gram block; cyclic method ordering balances positions
exactly over six repetitions. The two greedy blocks are separate contemporaneous
baselines. Every decoding method shares the target, inputs, cache and FP32 settings.
Any token disagreement aborts the run. Existing output directories are never replaced.
If a run fails, completed child artifacts remain, but no complete aggregate manifest
is written; do not interpret an incomplete directory as a completed replication.

The primary speedup is the median across cases of each case's median paired ratio.
Conditional bootstrap intervals resample whole cases, keeping all six repetitions
together. Results retain every dataset/training/task combination, target accuracy,
slowdown, raw timing, acceptance, target-call count and draft overhead. Dataset-level
ranges describe three independent datasets; timing repeats do not increase that count.

```bash
.venv/bin/specdecode verify-replication results/cpu-dataset-replication
```

Offline verification regenerates and checks all datasets and split disjointness,
replays all nine checkpoints on all held-out cases, checks raw timing pairing and
position balance, and recomputes the case bootstrap and aggregate reports. It does
not rerun timings. Exact data live in shared compressed dataset shards; correctness
is sharded by task and timings by draft length, as deterministic gzip JSONL. Source
hashes must match the frozen protocol to start a fresh run; verification reports
source differences without hiding them. The isolated installation check below also
replays the pilot and all nine final targets from outside the source tree.

## Replicate across seeds and sizes

The [six-configuration comparison](results/cpu-multiseed/RESULTS.md) extends the original single-target experiment. Run the complete design with one command:

```bash
.venv/bin/specdecode suite --output .runs/replication --seeds 17 29 43 --widths 64 96 --data-seed 2026 --train-per-task 256 --eval-per-task 12 --steps 800 --batch-size 32 --learning-rate 0.003 --threads 1 --warmups 2 --repeats 9 --draft-lengths 1 2 4 8 --max-new-tokens 25
```

All options above are the suite defaults. Both widths use two transformer layers, four heads, feed-forward width 4×model width, context 64, vocabulary 22, FP32, and one inter-op thread. The six training budgets are bounded at 800 steps each. Training initialization and minibatch seeds vary; the entire training/evaluation dataset is fixed and byte-identical across configurations. Each held-out prompt is disjoint from training. Sizes are interleaved in execution order. Dataset and checkpoint hashes, hardware, dependencies, and training costs are saved individually.

The suite uses a seeded random method permutation per case, cyclically rotated across nine measured repetitions. Every method occupies every timing position once per case; case order is shuffled. Repeats must be a multiple of the number of methods. This balances positions, not every possible predecessor/carryover effect. The legacy `run` command still defaults to five shuffled repetitions.

The output holds six self-contained experiments plus `suite.json`, `suite-report.json`, a generated `RESULTS.md`, and a manifest linking all child manifests. The cross-configuration report includes every method's speedup for each seed/size and separate target accuracy; individual reports retain latency, acceptance, target calls, and drafting cost. No settings or slowdowns are filtered out. Repeated timing samples do not increase the number of independent trained targets. Fixed optimizer steps do not equalize training FLOPs or quality between sizes.

```bash
.venv/bin/specdecode verify-suite results/cpu-multiseed
```

This verifies the manifest chain, regenerates all inputs, replays all six checkpoints' cache audits and token agreement, checks complete paired samples and balanced timing positions, replays task accuracy, and recomputes individual and cross-configuration summaries. SHA-256 checks detect accidental corruption, not malicious replacement of artifacts and manifests together. Verification does not rerun timing measurements. Use a new output directory for replication; existing artifacts are never overwritten. A failed run leaves its completed child artifacts available for individual verification.

## Historical synthetic methods

All three methods use **the same target, FP32 numerical settings, attention implementation, and KV cache**. There is no uncached performance baseline hidden in the comparison. The uncached implementation is used only as a correctness oracle outside timing.

| Method | Draft source |
|---|---|
| Greedy | No draft; one new token per target call |
| Prompt lookup | Longest matching suffix of up to four tokens in earlier committed history; copy the earliest match's known continuation |
| Learned n-gram | Training-only counts, up to four tokens of context, longest-context backoff to a unigram, deterministic smallest-token tie breaking |

Default draft lengths are 1, 2, 4 and 8. “Prompt lookup” includes already generated tokens, as in the existing method. Both drafters are immutable during evaluation. The n-gram table is rebuilt deterministically from the saved training split.

The decoder leaves the last committed token pending. One target call processes it together with the proposed block, comparing each proposal with the corresponding target argmax. At the first mismatch it emits the target's correction and discards the speculative cache tail. Full acceptance permits one bonus token. Both baseline and speculative methods prefill the prompt except its final token once. The shared prefill call is included in reported forward calls and latency. See [the decoder](src/specdecode/decode.py).

Generation stops on EOS, the output limit, or when the committed sequence reaches the model's context length (64). There is no sliding context window. Drafts reserve room for a correction/bonus token and never exceed the remaining context. Tied target logits use PyTorch's first-index argmax. Batched and single-token matmuls can round differently: every run checks actual token agreement and **fails on any disagreement**, rather than silently correcting its benchmark outputs.

## Synthetic tasks and target quality

Every sequence is `BOS, task ID, 48 payload tokens, EOS`. Prompts contain 24 payload tokens; the continuation contains another 24 plus EOS. The vocabulary has 22 tokens.

| Task | Payload | Purpose |
|---|---|---|
| Repetitive | A uniformly drawn four-symbol motif repeated 12 times | Strong history-copy opportunity; held-out motifs |
| Structured | 16 records of random key, deterministically mapped value, separator | Predictable fields interspersed with unpredictable choices |
| Unpredictable | 48 independent uniform symbols from a 16-symbol alphabet | A low-predictability control |

Training and evaluation have separate seeded PRNG streams. Evaluation prompts and full sequences are explicitly excluded from training; prefixes are unique within each split. This is an IID synthetic holdout, not a claim of compositional or natural-language generalization. Default sizes are 256 training and 12 evaluation sequences per task. The target has two pre-norm transformer layers, width 64, four heads, learned absolute positions, a 256-wide feed-forward layer, and no dropout. It trains for 800 AdamW steps with batches of 32; all settings and sampled training-loss checkpoints are saved.

Accuracy is reported separately from decoding agreement. Teacher-forced accuracy measures next-token prediction given true prefixes. Free-running accuracy compares generated tokens with the entire true continuation (missing tokens count as wrong); exact continuation requires the whole sequence including EOS. Payload-only teacher-forced accuracy excludes EOS. Random continuations cannot be recovered from their prompts: an ideal predictor achieves expected payload accuracy of 1/16 on the unpredictable task and 17/24 on the structured task under teacher forcing. Greedy generation may collapse to repetitive output that is easy to draft but has poor task accuracy. Speedups on that output are not evidence of useful random-sequence prediction.

## Measurement and artifacts

Each case/method is warmed twice. Five measured repetitions pair methods by evaluation case and repetition; both case order and method order are shuffled with a saved seed. Timing uses `perf_counter_ns` around the entire decoding call, including prompt prefill, tensor creation, drafting, rollback, instrumentation and Python overhead. Training, n-gram fitting, artifact I/O, correctness audits, and warmups are excluded. Draft overhead is instrumented inside those same calls; it includes proposal construction, but not the decoder's subsequent proposal validation/clipping. Batch size is one for decoding.

Summary speedup is the median of **paired** greedy/method latency ratios. The report also records the ratio of summed paired latencies and min/max ratios; these are descriptive statistics, not confidence intervals. No “best of” samples are discarded. Acceptance is accepted draft tokens / proposed draft tokens, including proposals discarded after an earlier rejection. Greedy has no acceptance denominator. Forward-token counts include wasted speculative inputs and prefill. Raw samples retain generated-token counts, forward calls, forward tokens, accepted/proposed tokens, rejected and fully accepted blocks, drafting time, stop reason, pairing and execution order.

| Artifact | Contents |
|---|---|
| `dataset.json` | Exact training/evaluation inputs, task labels, prefix lengths, split seeds |
| `target.pt` | Target weights and model configuration; loaded with `weights_only=True` |
| `config.json` | All run settings and seeds |
| `correctness.json` | Ground truth, greedy outputs, per-method outputs/counters, per-case accuracy |
| `timings.jsonl` | Every raw measured sample |
| `report.json` | Summaries, training trace/cost, draft fitting cost, hardware, affinity, threads, dependencies, source hashes and checkpoint hash |
| `RESULTS.md` | Generated readable timing and accuracy tables |
| `manifest.json` | SHA-256 of the other artifacts |

`verify` checks artifact hashes, regenerates the exact dataset, loads the saved model, and replays cache audits and token agreement for every method/case. It also reports whether the current source hashes match the original run. The checkpoint hash identifies the actual saved weights; identical retraining bytes across PyTorch builds or machines are not promised. Source-file hashes and package versions make implementation/environment differences inspectable. Do not compare raw timings across different machines as if the hardware were controlled.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_install.py
```

The second command builds a wheel, installs it and its public CPU dependencies in a fresh virtual environment, runs the tests from outside the source tree, and exercises CLI training, benchmarking, hash verification, re-timing, a small six-target suite, and replay of the recorded full suite. It requires internet for installation and `pip >= 22.3` in the invoking environment. Temporary installation files are removed when it finishes.

Tests cover rejection at each block position, complete acceptance/bonus generation, no-match fallback, accepted/rejected EOS, one-token and zero-token budgets, full/near-full contexts, corrupted-artifact rejection, dataset disjointness, paired sample accounting, timing-position balance, six-target shared-data identity, cross-report tampering, and FP32/FP64 transformer cache rollback against an uncached reference. The small integration test trains a real model and produces/replays all artifacts. These are empirical checks, not a proof against every possible floating-point argmax divergence.

## Historical synthetic study: existing work and limits

This is a reproduction-oriented educational experiment, not a novel decoding method. [Leviathan, Kalman and Matias (ICML 2023)](https://proceedings.mlr.press/v202/leviathan23a.html) introduced exact speculative sampling; this project implements only deterministic greedy verification, not their sampling acceptance rule. [Apoorv Saxena's prompt lookup implementation](https://github.com/apoorvumang/prompt-lookup-decoding) uses suffix matches to draft from existing token history. [llama.cpp's lookup example](https://github.com/ggml-org/llama.cpp/tree/master/examples/lookup) provides an existing production-oriented lookup implementation. This repository implements its own small model and verifiers to expose cache behavior and measurement costs.

Conclusions are limited to the saved CPU, thread count, tiny target, seed, synthetic generators, and measured cases. A shared virtual CPU can have noisy timings. Those synthetic models were not trained on natural language; the experiment does not establish production serving performance, GPU speedups, sampling equivalence, or benefits on large models. Poor task accuracy, draft rejection, and measured slowdowns are results to report, not failures to hide.
