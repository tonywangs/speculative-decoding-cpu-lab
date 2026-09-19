"""A fixed-data factorial comparison; no pooling of timing repeats as model trials."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .data import TASKS
from .experiment import RunConfig, prepare_output, run, sha256, verify, write_json


@dataclass(frozen=True)
class SuiteConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    widths: tuple[int, ...] = (64, 96)
    data_seed: int = 2026
    train_per_task: int = 256
    eval_per_task: int = 12
    steps: int = 800
    batch_size: int = 32
    learning_rate: float = 0.003
    threads: int = 1
    warmups: int = 2
    repeats: int = 9
    draft_lengths: tuple[int, ...] = (1, 2, 4, 8)
    max_new_tokens: int = 25

    def __post_init__(self):
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("suite requires three distinct training seeds")
        if len(self.widths) != 2 or len(set(self.widths)) != 2:
            raise ValueError("suite requires two distinct model widths")
        for width in self.widths:
            self.run_config(self.seeds[0], width)

    def run_config(self, seed, width):
        common = asdict(self)
        common.pop("seeds")
        common.pop("widths")
        return RunConfig(seed=seed, width=width, balanced_order=True, **common)

    def configurations(self):
        # Interleave sizes to avoid confounding size with the beginning/end of a run.
        return [(f"width{width}-seed{seed}", self.run_config(seed, width))
                for seed in self.seeds for width in self.widths]


def collect(path, cfg):
    configurations = []
    dataset_hashes = set()
    for name, run_cfg in cfg.configurations():
        child = path / name
        report = json.loads((child / "report.json").read_text())
        # JSON-normalize tuple/list fields before comparing saved settings.
        if report["config"] != json.loads(json.dumps(asdict(run_cfg))):
            raise AssertionError(f"suite configuration mismatch: {name}")
        if not report["agreement_passed"]:
            raise AssertionError(f"agreement failed: {name}")
        dataset_hashes.add(sha256(child / "dataset.json"))
        configurations.append({"name": name, "seed": run_cfg.seed, "width": run_cfg.width,
                               "parameter_count": report["training"]["parameter_count"],
                               "checkpoint_sha256": report["checkpoint_sha256"],
                               "evaluation_cases": report["evaluation_cases"],
                               "timed_samples": report["timed_samples"],
                               "summary": report["summary"], "accuracy": report["accuracy"]})
    if len(dataset_hashes) != 1:
        raise AssertionError("suite datasets differ")
    variation = []
    for width in cfg.widths:
        models = [c for c in configurations if c["width"] == width]
        for task in TASKS:
            for method in [r["method"] for r in models[0]["summary"] if r["task"] == task]:
                values = [next(r["paired_speedup_median"] for r in c["summary"]
                               if r["task"] == task and r["method"] == method) for c in models]
                variation.append({"width": width, "task": task, "method": method,
                                  "seed_speedups": dict(zip(map(str, cfg.seeds), values)),
                                  "min": min(values), "median": statistics.median(values), "max": max(values),
                                  "slowdown_seeds": [c["seed"] for c, v in zip(models, values) if v < 1]})
    return {"schema_version": 1, "dataset_sha256": dataset_hashes.pop(),
            "agreement_passed": True, "configurations": configurations, "variation": variation,
            "timed_samples": sum(c["timed_samples"] for c in configurations),
            "evaluation_cases": configurations[0]["evaluation_cases"]}


def markdown(report, cfg):
    lines = ["# Three-seed, two-size CPU comparison", "",
             f"Training seeds: {list(cfg.seeds)}; shared data seed: {cfg.data_seed}. "
             f"Widths: {list(cfg.widths)}; two layers, four heads, feed-forward width 4×model width, "
             "context 64, vocabulary 22. FP32 CPU, deterministic algorithms, cached decoding for every timed method.", "",
             f"Each target trains for {cfg.steps} AdamW steps (batch {cfg.batch_size}, learning rate {cfg.learning_rate}) "
             f"on {cfg.train_per_task} sequences per task. {cfg.eval_per_task} held-out cases per task; "
             f"{cfg.threads} intra-op and one inter-op thread; {cfg.warmups} warmups and {cfg.repeats} repetitions. "
             f"Draft lengths: {list(cfg.draft_lengths)}; output limit {cfg.max_new_tokens}.", "",
             "All six configurations use byte-identical training and evaluation data; all evaluation prefixes are "
             "excluded from training. Seeds vary initialization and minibatch sampling, not the dataset. "
             "One random method permutation per case is cyclically rotated across repetitions, balancing each "
             "method at every position within each case. Case order is shuffled. This balances position, not all "
             "possible carryover effects. Configurations run sequentially with sizes interleaved.", "",
             f"All {report['timed_samples']:,} measured outputs matched their target's uncached greedy reference. "
             "Different targets need not produce the same outputs. Warmups and correctness audits also require agreement.", "",
             "## Variation across training seeds", "",
             "Each cell is a within-case paired latency speedup, summarized separately per trained target, then "
             "described across three seeds. Values below 1 are slowdowns. No samples or slow settings are omitted. "
             "The range is descriptive, not a confidence interval; timing repeats are not independent model trials.", "",
             "| Width | Task | Method | " + " | ".join(f"Seed {s}" for s in cfg.seeds) + " | Median | Min–max | Slowdown seeds |",
             "|---:|---|---|" + "---:|" * len(cfg.seeds) + "---:|---|---|"]
    for r in report["variation"]:
        values = " | ".join(f"{r['seed_speedups'][str(s)]:.3f}×" for s in cfg.seeds)
        lines.append(f"| {r['width']} | {r['task']} | {r['method']} | {values} | {r['median']:.3f}× | "
                     f"{r['min']:.3f}–{r['max']:.3f} | {', '.join(map(str, r['slowdown_seeds'])) or '—'} |")
    lines += ["", "## Target quality and individual measurements", "",
              "Accuracy is against the same held-out truths, independently of decoding agreement. Token accuracy "
              "includes EOS except the payload column. Missing generated tokens count as wrong. Unpredictable "
              "continuations cannot be inferred from their prompts; repetitive model output can be fast to draft "
              "despite low task accuracy.", "",
              "| Configuration | Parameters | Task | Teacher-forced | Payload only | Free-running | Exact continuation |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for c in report["configurations"]:
        for r in c["accuracy"]:
            lines.append(f"| [{c['name']}]({c['name']}/RESULTS.md) | {c['parameter_count']:,} | {r['task']} | "
                         f"{r['teacher_forced_token_accuracy']:.1%} | {r['teacher_forced_payload_accuracy']:.1%} | "
                         f"{r['free_running_token_accuracy']:.1%} | {r['exact_continuation_accuracy']:.1%} |")
    lines += ["", "Each linked report contains latency, acceptance, target calls and drafting overhead for every "
              "task/method. Adjacent files preserve all raw timings, inputs, correctness records, weights, hashes, "
              "training traces, dependency versions and hardware details. suite-report.json preserves unrounded "
              "per-configuration results and across-seed ranges.", "",
              "## Limits", "",
              "Only three initializations, two tiny widths, one shared synthetic dataset, one CPU environment "
              "and greedy decoding were tested. Width changes also change parameter count and training compute; "
              "the step budget, not training FLOPs or achieved quality, is held fixed. This does not isolate model "
              "size from output quality or draftability. Small held-out sets and shared-host scheduling limit "
              "precision. No extrapolation to large models, natural language, GPUs or sampling is justified. "
              "There is no novelty claim: this extends the existing greedy speculative verification, prompt "
              "lookup and n-gram baseline experiment with replication.", ""]
    return "\n".join(lines)


def run_suite(output, cfg):
    path = prepare_output(output)
    write_json(path / "suite.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg)})
    for name, run_cfg in cfg.configurations():
        print(f"Running {name}...", flush=True)
        run(path / name, run_cfg)
    report = collect(path, cfg)
    write_json(path / "suite-report.json", report)
    (path / "RESULTS.md").write_text(markdown(report, cfg))
    names = ["suite.json", "suite-report.json", "RESULTS.md"] + [f"{name}/manifest.json" for name, _ in cfg.configurations()]
    write_json(path / "manifest.json", {name: sha256(path / name) for name in names})
    return report


def verify_suite(path):
    path = Path(path)
    cfg = SuiteConfig(**json.loads((path / "suite.json").read_text())["config"])
    expected = {"suite.json", "suite-report.json", "RESULTS.md"} | {f"{name}/manifest.json" for name, _ in cfg.configurations()}
    manifest = json.loads((path / "manifest.json").read_text())
    if set(manifest) != expected or any(sha256(path / name) != digest for name, digest in manifest.items()):
        raise ValueError("suite artifact hash mismatch")
    verified = {name: verify(path / name) for name, _ in cfg.configurations()}
    report = collect(path, cfg)
    if report != json.loads((path / "suite-report.json").read_text()):
        raise AssertionError("suite report does not match individual runs")
    if (path / "RESULTS.md").read_text() != markdown(report, cfg):
        raise AssertionError("suite markdown does not match report")
    return {"artifact_hashes_verified": True, "agreement_passed": True, "summary_recomputed": True,
            "identical_datasets": True, "configurations": verified, "timed_samples": report["timed_samples"]}
