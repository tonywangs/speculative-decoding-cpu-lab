# Resumable research calibration

This controller measures the cost of a proposed two-experiment study. It currently
executes **only six calibration units**, then stops. The full autonomous
validation → decision → test study is unfinished. Neither experiment has been
completed over the population.

The saved [calibration results](../results/controller-calibration/RESULTS.md) use
the original two target/GRU pairs and pinned Shakespeare, Carroll and CPython
bytes. No models are trained, downloaded, or served remotely.

## Inspect the saved run

From the project root with the installed local environment:

```bash
.venv/bin/python -m specdecode.research status
.venv/bin/python -m specdecode.research verify
```

`status` shows durable budgets, completed records and the exact next operation.
`verify` checks the source/model identities, complete window enumeration,
measurement accounting, output agreement, balanced positions and exports. It
uses saved evidence; it does not repeat inference or timing.

## Start or resume a calibration

Choose a new output directory and an absolute deadline that you are authorized
to use. The process stops two minutes before that deadline. Replace the timestamp
below; it is a placeholder, not a schedule:

```bash
.venv/bin/python -m specdecode.research init --output .runs/my-calibration
.venv/bin/python -m specdecode.research calibrate --output .runs/my-calibration --deadline YYYY-MM-DDTHH:MM:SS+00:00
```

Run the same `calibrate` command to resume that directory. `--max-units 1` pauses
between units; `--stop-after-records 10` pauses after ten newly completed
operations. SIGINT/SIGTERM also save an interrupted attempt. A completed
calibration is idempotent: another invocation performs no model forwards. There
is no recurring job, daemon, full-study command, or automatic budget increase.

Linux is required. The runner uses one process and no subprocess workers.
Defaults are immutable in the journal: six calibration units, 600 cumulative
execution seconds, 150,000 aggregate target + GRU forward invocations, a 4 GiB
address-space ceiling, zero external calls, and two **pending** full experiments.
The address-space ceiling is stricter than a resident-memory ceiling. Peak
resident memory is measured with `getrusage`; with no child workers it is also
the process-tree peak. Imports and derived JSON export are outside the execution
budget; model loading, fitting, source checks, inference and per-operation
checkpoint/orchestration time are included. Export cost is separately logged.

Before each operation the journal reserves up to 30 seconds and 512 calls from
the remaining cumulative allowance. An interval timer stops the operation at its
lease; every model forward also checks time and calls. The call count includes
uncached references, full-prefix numerical audits, quality passes, target
prefill/verification, and every GRU invocation. Signal delivery waits for a native
kernel to return; cleanup and durable stop-record writes can occur after the
clock boundary. No subsequent operation is launched after exhaustion.

## Population and frozen proposals

The manifest retains all 831 eligible validation and 832 eligible test windows,
both seeds, original training windows, source/model hashes, split bounds and the
excluded window. Windows are nonoverlapping 160-byte segments: 96 prompt bytes
plus 64 continuation bytes. An initial BOS token is retained. Low-diversity
prompts, duplicated prompts, and any prompt found in its entire source training
region are excluded. Exactly one Python validation window is excluded here.

The calibration takes the first eligible validation window in ascending source
offset for each source and seed. These six units never replace the population.
Each unit has 73 operations:

- One independent uncached-greedy reference and two teacher-forced quality passes.
- Seven numerically audited decodes, each checked against that reference.
- Two warmups for each of seven configurations.
- Seven paired timing repetitions; cyclic permutations put each configuration
  in each position exactly once. Every output is checked again.

Configurations are cached greedy plus fixed k2 and acceptance-adaptive k2/k4 for
prompt lookup, training-only n-grams, and the saved learned GRU. The adaptive rule
starts at k2 for each request, promotes after two consecutive fully accepted
nonempty proposal blocks, and demotes on rejection. The trace records every
policy update. The timer includes reset, prefill, draft generation, verification,
rollback, policy updates, accounting hooks and Python request overhead. Equality
checking, serialization and journal writes are outside request latency and
included separately in controller execution cost.

The second experiment is predeclared in `proposal-policy.json`. Only **complete
validation** can select it: if the adaptive GRU's median across case-level paired
speedups is below one for either seed, select a per-request fallback after four
proposal blocks with cumulative acceptance below 0.5. Otherwise select a stricter
four-block promotion threshold. Both branches retain all three draft families
and fixed-k2 controls. Both choices must be frozen before full test evaluation.
The selection function rejects incomplete, duplicate, incorrect, or test-split
evidence. It has not selected a branch using this calibration.

## Durability and interpretation

`journal.sqlite3` is authoritative. Work and metadata are immutable; measurements
and events are append-only. A result, its call counts and the unused-budget refund
commit atomically. A killed process leaves a reservation: recovery charges that
reservation in full, logs the loss, and keeps the work pending. It never pretends
to know the exact calls lost in a crash. Recovered work has a new attempt and can
produce only one successful result. An exclusive OS lock prevents simultaneous
workers for a journal. Copying a run deliberately creates a separate run; the
controller does not coordinate across unrelated output directories.

If a process restarts inside a timing unit, it performs two fresh warmups for all
methods before further timings. Those additional decodes are explicitly logged
as `resume_warmup`, consume the same budgets, and do not replace or duplicate
successful measurements. Small-model integration tests exercise this path and
SIGTERM recovery; they are mechanism tests, not scientific evidence.

`population.json`, `proposal-policy.json`, `state.json`, `measurements.jsonl` and
`extrapolation.json` are derived views. The journal commits every operation, well
inside the five-minute checkpoint interval; views are refreshed on orderly exit.
A hard-killed run can therefore have stale views until it is resumed. Do not edit
the journal or treat a stale JSON file as its source of truth.

The extrapolation weights each calibration source/seed by its complete source
population and accounts for audits, quality, warmups, timings and observed
controller overhead. Oracles and quality are shared between the two experiments.
Both possible second-experiment branches get an explicitly labeled cost analogue
and a conservative forward-call upper bound. They have not been measured, so
neither has a proven wall-time bound. Six windows cannot establish population
speedups, forecast precision, or useful language quality.
