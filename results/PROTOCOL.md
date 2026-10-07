# Frozen dataset replication protocol

This experiment tests whether the earlier synthetic CPU findings persist when the
dataset changes, at one fixed target size. It is a replication, not a new decoding
algorithm. The nine final configurations are specified by
[cpu-dataset-replication/protocol.json](cpu-dataset-replication/protocol.json).

## Existing work reviewed before final evaluation

Reviewed on 2026-09-27:

- [Leviathan, Kalman and Matias, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)
  describes speculative sampling that preserves the target distribution. This
  project tests only deterministic greedy verification, not that sampling rule.
- [Apoorv Saxena's prompt lookup implementation](https://github.com/apoorvumang/prompt-lookup-decoding)
  drafts continuations from matching token history. The local implementation uses
  longest suffix lookup with deterministic selection, and includes generated history.
- The local historical [six-target experiment](cpu-multiseed/RESULTS.md), decoder,
  n-gram implementation and cache regressions were inspected and replayed. Their
  implementations and recorded artifacts are retained. This extension reuses the
  same target, training, drafter, correctness and timing routines.

The historical data suggested strong prompt-lookup gains for repetition and
slowdowns on unpredictable inputs. The new design checks those directions without
selecting final seeds, prompts or settings based on final results.

## Development-only pilot

The [pilot artifacts](cpu-dataset-pilot/experiment/RESULTS.md) use dataset seed
11003 (evaluation stream 11004), training seed 7, width 64, 800 training steps,
256 training sequences per task, eight held-out examples per task and four timed
examples per task. Its data and training seed are excluded from the final design.
Both candidate lengths, 2 and 8, were retained; no search over final data was run.

The single pilot trained for **60.888 seconds** (63.902 seconds including setup),
and the complete configuration took **93.656 seconds**. All 24 audit cases and
432 timed outputs agreed with uncached greedy. The pilot artifacts occupy 524,622
bytes including the duplicated input protocol. These are measured development
values, not final-run measurements or a performance claim. Pilot training partly
overlapped the end of the small regression suite, so its training cost is only a
rough planning estimate. Its measurement phase ran after that suite finished.

The larger final audit and timing sets should fit comfortably within a two-hour
training/measurement cap based on this pilot. The actual final invocation uses
`timeout --signal=TERM 2h` around the reproduction command. This is a one-shot
elapsed-time cap, not a schedule. Any timeout or mismatch leaves an incomplete run;
no missing measurements may be filled in or a failing seed replaced. Replay and
installation validation run after measurement so they do not compete with it.

## Final design, fixed before evaluation

- Dataset seeds: 3101, 6203, 9307. Each dataset uses separate training and evaluation
  PRNG streams (`seed` and `seed + 1`), with no shared stream seed across datasets.
- Paired initialization seeds: 17, 29, 43. Minibatch PRNG seed is initialization
  seed + 2. Identical index-sampling rules are used across datasets.
- One model size: width 64, two layers, four heads, feed-forward width 256,
  context 64, vocabulary 22; 107,008 parameters, FP32 CPU, deterministic PyTorch,
  one intra-op and one inter-op thread. Existing cached decoder for every method.
- Training: 256 sequences per task, 800 AdamW steps, batch 32, learning rate .003,
  weight decay .01, betas .9/.999, epsilon 1e-8, gradient norm clip 1.0; no scheduler.
- Accuracy and agreement: **128 held-out examples per task per target**, all
  384 cases, with uncached greedy as the token oracle and cache rollback audits.
  Prompt prefixes are unique and explicitly disjoint from training prefixes.
- Timing: **first 16 sorted evaluation IDs per task**, fixed without looking at
  predictions or timings; 48 cases per target. Two warmups and six repetitions.
- Lengths: **2 and 8**. Each length gets a separate three-method block (greedy,
  prompt lookup, training-only n-grams), including its own paired greedy baseline.
  The same immutable drafters are used for audit and measurements. Output limit 25.
- Within a block, a seeded random base permutation per case is cyclically rotated;
  over six repetitions every method occupies each position twice. This balances
  position, not every possible predecessor effect. Cases are shuffled each repeat.
  Order seed is training seed + 3. Length order alternates across configurations.
- Configuration order rotates datasets across training seeds, recorded explicitly
  in protocol.json. Configurations run sequentially, never concurrently.
  Length-order coverage is uneven by dataset: seed 9307 measures length 2 before
  length 8 for all three training seeds. Each length has its own paired greedy
  baseline; the design does not separate all block carryover effects.

Canonical dataset SHA-256 hashes and exact timing IDs are frozen in protocol.json,
alongside source hashes, dimensions and analysis rules. Freeze refuses to overwrite
an existing protocol. Execution rejects source drift after freezing. The final run
has 7,200 training steps, 3,456 audited case/target combinations (128 × 3 × 9),
and 15,552 timed calls (16 × 3 × 3 methods × 6 repeats × 2 lengths × 9 targets).
Every audit, warmup and timed output must match that target's ordinary greedy output.

## Analysis fixed before evaluation

The primary speedup for each configuration/task/method is the median across timed
cases of the case's median of six paired greedy/method latency ratios. Ratios below
one mean slowdown. Every setting, including slowdowns, is retained. The legacy
median over all paired samples and ratio of summed latencies are also preserved in
child reports, explicitly separate from the new primary case-level estimate.

Conditional uncertainty uses 2,000 percentile-bootstrap draws of whole cases,
resampling all six paired repetitions together. PRNG seed 701 is reset per
configuration/task/method. The 2.5th and 97.5th percentiles use linear interpolation.
These intervals describe timed-case variability for a fixed trained model and
dataset. They are not intervals over unseen datasets, training seeds, machines or
host conditions. Only 16 cases per task are timed.

For each dataset, report all three training estimates and their geometric mean.
Describe the minimum, median and maximum of the three dataset geometric means;
these ranges are not confidence intervals. Do not pool timing repetitions as
independent datasets, and do not select a best seed or length. With only three
datasets, population-level inference would be fragile, so no significance or
natural-language generalization is claimed.

Accuracy is reported separately using all 128 held-out examples per task: teacher-
forced token and payload accuracy, free-running accuracy and exact continuation.
EOS is included except in the payload metric; missing output tokens count as wrong.
Raw timing shards preserve latency, acceptance, forward calls/tokens, draft overhead,
stop reason and execution positions. Training trace, checkpoint hashes, source
hashes, package versions and CPU details remain in each target report.

## Reproduction and verification

After the README's installation steps:

```bash
.venv/bin/specdecode replicate --protocol results/cpu-dataset-replication/protocol.json --output .runs/dataset-replication
.venv/bin/specdecode verify-replication results/cpu-dataset-replication
.venv/bin/python scripts/verify_install.py
```

Use an empty output directory. Compressed JSONL shards are deterministic gzip with
mtime zero; exact datasets are stored once per dataset seed and shared by its three
targets. Each target's `config.json` describes training and the full five-method audit; its
`balanced_order=false` is not the timing-block setting. Timing configurations are
derived from the frozen protocol with one draft length and `balanced_order=true`,
and raw positions are checked separately for each three-method block.
The offline verifier regenerates the data, checks splits and manifest
chains, loads every checkpoint with `weights_only=True`, replays every audit,
checks complete pairing/counters/position balance, and recomputes all summaries.
The installation check builds a wheel in a clean environment, runs outside the
source tree, and replays historical, development and final checkpoints. Downloads
are needed for that installation; checkpoint replay itself uses no network.

## Limits

This covers three synthetic datasets and three paired training seeds, one tiny
model size and one shared CPU environment. Training quality can change draft
acceptance even with the same training budget. The structured and unpredictable
truths contain random continuations that cannot be predicted from their prompts;
fast repetitive model output need not be correct. Conditional bootstrap intervals
exclude important sources of uncertainty. No conclusions about natural-language
models, GPUs, stochastic decoding or production serving follow from this experiment.
