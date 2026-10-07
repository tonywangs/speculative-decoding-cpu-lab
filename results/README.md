# Recorded experiments

## Replication across datasets and paired training seeds

The [nine-target dataset replication](cpu-dataset-replication/RESULTS.md) tests one
fixed width-64 model on dataset seeds 3101/6203/9307 and training seeds 17/29/43.
The [protocol](PROTOCOL.md) was frozen after a separate development pilot. Each
target was audited on 128 held-out examples per task and timed on a predefined
16-example subset per task, with two warmups and six balanced paired repetitions
at draft lengths 2 and 8. All **3,456 case/target audits** and **15,552 timed outputs**
matched uncached greedy. There are 1,152 unique held-out inputs shared across the
three training seeds, not 3,456 independent inputs or nine independent datasets.

The main historical directions persisted in the observed medians:

- Length-8 prompt lookup on repetition achieved **5.140–5.373×** across the nine
  targets. Dataset geometric means across training seeds were **5.311×, 5.250×,
  and 5.203×**, respectively. Repetition free-running accuracy was **97.31–98.97%**,
  while exact-continuation accuracy was **70.31–88.28%**.
- Length-8 n-grams on structured records achieved **2.236–2.692×**. Structured
  free-running accuracy was only **42.41–44.94%** and exact continuation was zero;
  those speedups do not recover unpredictable future keys.
- Both n-gram settings slowed repetition in every observed target median:
  **0.830–0.971×**. Six of these 18 conditional case-bootstrap intervals included
  parity, so not every individual slowdown is clearly resolved by the timed cases.
- All 36 speculative task/method/target medians on unpredictable data were below
  parity: **0.863–0.947×**. One conditional interval (prompt lookup, length 2)
  included parity. Payload teacher-forced accuracy was **5.60–7.00%**, near the
  6.25% chance level; no exact continuation was correct.

These are ranges of case-level speedup estimates, not pooled timing observations.
The full report keeps every task/dataset/training-seed result, conditional 95%
whole-case bootstrap intervals, accuracy, slowdowns and dataset-level descriptive
ranges. The intervals condition on a fixed target and dataset; they do not quantify
population uncertainty from only three datasets or shared-host timing effects.
No large-model, natural-language or stochastic-decoding conclusion follows.

All nine distinct checkpoints, three exact dataset shards, raw timings, correctness
records, hashes, versions and hardware details are preserved. The sum of recorded
configuration durations was **2,819.628 seconds (47.0 minutes)**, including training,
audit, warmups, timing and analysis; training loops alone summed to **509.001
seconds**. These per-configuration intervals exclude final report/manifest writes
and top-level aggregation. The run completed within its two-hour cap. See the main README for the
single reproduction command and the frozen protocol for exact analysis rules.

```bash
.venv/bin/specdecode verify-replication results/cpu-dataset-replication
```

## Earlier single-dataset comparison

The [three-seed, two-size comparison](cpu-multiseed/RESULTS.md) is the earlier replication across initializations and sizes. It uses identical data across all six targets, nine balanced timing repetitions, and retains every setting including slowdowns. Each configuration has its own complete raw artifacts. See the [main README](../README.md#replicate-across-seeds-and-sizes) for the explicit reproduction command.

```bash
.venv/bin/specdecode verify-suite results/cpu-multiseed
```

The six targets contain 107,008 (width 64) or 234,240 (width 96) parameters. All 17,496 timed outputs matched each target's uncached greedy reference. Across the six trained targets:

- Length-8 prompt lookup on repetition achieved 4.815×–5.196× median paired speedup. Free-running repetition accuracy ranged from 95.3% to 99.7%.
- Length-8 n-grams on structured records achieved 2.168×–2.479×. Structured free-running accuracy remained only 40.7%–47.3%, despite teacher-forced payload accuracy of 69.1%–73.3%.
- Every n-gram setting slowed repetition (0.805×–0.894× across configurations and lengths).
- Every speculative setting slowed the unpredictable task (0.815×–0.961×). Its teacher-forced payload accuracy was 5.2%–7.6%, near the 6.25% chance level; free-running accuracy including EOS was 7.0%–10.0%.

These are ranges of per-target summary statistics, not confidence intervals. Only initialization/minibatch seeds and width vary: the single dataset, CPU setting, and step budget are shared. A wider target receives more training compute per step, and target outputs differ, so the comparison does not isolate a causal effect of parameter count. Full tables retain all settings and task-quality results, including slowdowns. No large-model or natural-language conclusion follows.

## Original single-seed experiment

[cpu-seed17/RESULTS.md](cpu-seed17/RESULTS.md) contains the full comparison, with raw artifacts beside it. It was generated by the default command:

```bash
.venv/bin/specdecode run --output results/cpu-seed17
```

Use a different empty output directory to reproduce it. The run uses seed 17, 768 training sequences, 36 disjoint evaluation prompts, 800 training steps, draft lengths 1/2/4/8, two warmups per method/case, and five measured repetitions: 1,620 timing samples. The target has 107,008 parameters and uses one intra-op and one inter-op thread on a four-vCPU `DO-Regular` Linux host. All outputs matched the uncached greedy reference. The environment, checkpoint hash and source hashes are recorded in `report.json`.

The central observation is that drafting depends on matching the **target's output**, not merely on recognizing that a task is repetitive or structured:

- Prompt lookup at length 8 achieved a 5.149× median paired speedup on repetition. The motif is already present several times in the prompt. Calls fell from 26 to 4.42 on average, including prefill; 90.6% of proposed tokens were accepted. Target teacher-forced and free-running accuracy were both 98.7%.
- Learned n-grams at length 8 achieved 2.425× on structured records, with 31.4% acceptance. Many value/separator fields are predictable from training. Teacher-forced accuracy was 70.7%, but free-running accuracy was only 42.7% because random keys cannot be inferred from the prompt.
- Learned n-grams at lengths 2/4/8 slowed the repetitive task down (0.931×/0.888×/0.868×). Prompt lookup can copy the current motif directly; the training-only n-gram table instead relies on continuations learned from other sequences. Length 1 was approximately parity at 1.012×; this small difference should not be treated as a robust gain on a shared host.
- Every speculative setting slowed the unpredictable task down (0.861×–0.937×), despite modestly reducing target-call counts. Acceptance was low and block verification/drafting overhead exceeded the saved calls. Teacher-forced payload accuracy was near the 1/16 chance level; EOS raises the accuracy reported over the full continuation.

These are descriptive results from one target seed and 12 prompts per task. Repetitions measure timing variation, not independent training outcomes. Min/max ratios and every raw sample are retained in JSON; no confidence interval or population-level significance is claimed. The fixed context, small vocabulary, explicit Python/PyTorch implementation and virtual CPU are important limits. This result neither establishes the same gains for large models nor tests stochastic speculative sampling.

Replay the artifacts with:

```bash
.venv/bin/specdecode verify results/cpu-seed17
```
