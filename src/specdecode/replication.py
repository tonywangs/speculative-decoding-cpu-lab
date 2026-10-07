"""Dataset replication, kept separate from the historical experiment protocol."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
from time import perf_counter_ns

import torch

from .data import TASKS, generate_data
from .experiment import (RunConfig, configure, correctness, environment, measure,
                         methods, prepare_output, sha256, summarize, train,
                         validate_measurements, write_json)
from .model import ModelConfig, TinyTransformer


@dataclass(frozen=True)
class ReplicationConfig:
    data_seeds: tuple[int, ...] = (3101, 6203, 9307)
    training_seeds: tuple[int, ...] = (17, 29, 43)
    width: int = 64
    steps: int = 800
    train_per_task: int = 256
    eval_per_task: int = 128
    timed_per_task: int = 16
    batch_size: int = 32
    learning_rate: float = 0.003
    threads: int = 1
    warmups: int = 2
    repeats: int = 6
    draft_lengths: tuple[int, ...] = (2, 8)
    max_new_tokens: int = 25
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 701
    phase: str = "final"

    def __post_init__(self):
        if self.phase not in ("final", "development"):
            raise ValueError("unknown phase")
        if not self.data_seeds or not self.training_seeds:
            raise ValueError("nonempty seeds required")
        if len(set(self.data_seeds)) != len(self.data_seeds) or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("distinct seeds required")
        if len({s + offset for s in self.data_seeds for offset in (0, 1)}) != 2 * len(self.data_seeds):
            raise ValueError("dataset PRNG streams overlap")
        if not 1 <= self.timed_per_task <= self.eval_per_task or self.bootstrap_draws < 100:
            raise ValueError("invalid timing subset or bootstrap budget")
        if len(self.draft_lengths) != 2 or len(set(self.draft_lengths)) != 2:
            raise ValueError("two distinct draft lengths required")
        if self.phase == "final" and (len(self.data_seeds) != 3 or len(self.training_seeds) != 3
                                      or self.eval_per_task < 128 or self.timed_per_task < 16
                                      or self.repeats != 6):
            raise ValueError("final requires 3x3 seeds, >=128 audit/16 timing cases per task and six repeats")
        self.run_config(self.data_seeds[0], self.training_seeds[0])

    def run_config(self, data_seed, seed, k=None):
        # Each length has its own contemporaneous greedy baseline, three methods,
        # and six balanced paired repetitions. Full audits cover both lengths.
        return RunConfig(seed=seed, data_seed=data_seed, width=self.width,
                         steps=self.steps, train_per_task=self.train_per_task,
                         eval_per_task=self.eval_per_task, batch_size=self.batch_size,
                         learning_rate=self.learning_rate, threads=self.threads,
                         warmups=self.warmups, repeats=self.repeats,
                         draft_lengths=(k,) if k is not None else self.draft_lengths,
                         max_new_tokens=self.max_new_tokens, balanced_order=k is not None)

    def configurations(self):
        # Cross data and training seeds, rotate data order between training seeds.
        return [(f"data{d}-train{s}", d, s)
                for i, s in enumerate(self.training_seeds)
                for d in list(self.data_seeds)[i % len(self.data_seeds):] + list(self.data_seeds)[:i % len(self.data_seeds)]]


def compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(obj):
    return hashlib.sha256(compact(obj)).hexdigest()


def write_gzip(path, rows):
    # Deterministic compression, reproducible task/length shards under size limits.
    data = b"".join(compact(row) + b"\n" for row in rows)
    path.write_bytes(gzip.compress(data, mtime=0))


def read_gzip(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def audit_splits(data):
    train_prefixes = {tuple(r["tokens"][:r["prompt_length"]]) for r in data["train"]}
    eval_prefixes = {tuple(r["tokens"][:r["prompt_length"]]) for r in data["eval"]}
    if len(train_prefixes) != len(data["train"]) or len(eval_prefixes) != len(data["eval"]) or train_prefixes & eval_prefixes:
        raise AssertionError("nonunique or overlapping split prefixes")


def timing_cases(data, cfg):
    # Select on IDs only, before observing model outputs or timings.
    return [row for task in TASKS for row in sorted(
        (r for r in data["eval"] if r["task"] == task), key=lambda r: r["id"])[:cfg.timed_per_task]]


ANALYSIS = {
    "timing": "Separate three-method block per draft length; independent contemporaneous greedy baseline. Random base order per case, cyclic rotation over six repetitions; each method occupies each position twice. Length order alternates by configuration index.",
    "primary": "For each case, median of six paired greedy/method latency ratios; per-configuration estimate is median across timed cases. Values <1 are slowdowns; retain every sample.",
    "uncertainty": "Percentile 95% bootstrap interval resampling whole cases (all six pairs together), conditional on this fitted target and dataset. Fixed PRNG reset per task/method. No timing repetition is an independent dataset or training trial.",
    "across_seeds": "For each dataset report all three training estimates and their geometric mean; report min/median/max of the three dataset geometric means. These are descriptive ranges, not population confidence intervals. No pooling of raw repeats across datasets.",
    "accuracy": "All held-out cases, teacher-forced token/payload accuracy, free-running token accuracy, exact continuation; decoding agreement separately gated against uncached greedy.",
    "correctness": "Any token disagreement in audit, warmup or measurement aborts the configuration; never omit a failed configuration or substitute another seed.",
    "limits": "Three synthetic datasets and three paired training seeds, one tiny FP32 CPU model size, shared host. Conditional case bootstrap excludes training/dataset and host uncertainty. No natural-language, large-model, GPU or sampling generalization.",
}


def protocol_body(cfg):
    datasets = {}
    for seed in cfg.data_seeds:
        data = generate_data(seed, cfg.train_per_task, cfg.eval_per_task)
        audit_splits(data)
        datasets[str(seed)] = {"canonical_sha256": object_hash(data),
                               "timing_ids": [r["id"] for r in timing_cases(data, cfg)],
                               "train_count": len(data["train"]), "eval_count": len(data["eval"])}
    return {"schema_version": 1, "config": json.loads(json.dumps(asdict(cfg))),
            "model_config": asdict(ModelConfig(width=cfg.width)), "datasets": datasets,
            "configuration_order": [r[0] for r in cfg.configurations()], "analysis": ANALYSIS}


def freeze(path, cfg):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to replace frozen protocol")
    obj = protocol_body(cfg)
    obj["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    obj["source_sha256"] = environment()["source_sha256"]
    write_json(path, obj)
    return obj


def load_protocol(path):
    obj = json.loads(Path(path).read_text())
    cfg = ReplicationConfig(**obj["config"])
    expected = protocol_body(cfg)
    if {k: v for k, v in obj.items() if k not in ("frozen_utc", "source_sha256")} != expected:
        raise ValueError("protocol does not match frozen generation/analysis rules")
    return obj, cfg


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def case_estimates(raw, cfg):
    baseline = {(r["id"], r["repeat"]): r["elapsed_ns"] for r in raw if r["method"] == "greedy"}
    result = []
    for task in TASKS:
        for method in sorted({r["method"] for r in raw}):
            grouped = {}
            for r in raw:
                if r["task"] == task and r["method"] == method:
                    grouped.setdefault(r["id"], []).append(baseline[r["id"], r["repeat"]] / r["elapsed_ns"])
            values = {key: statistics.median(grouped[key]) for key in sorted(grouped)}
            samples = list(values.values())
            rng = random.Random(cfg.bootstrap_seed)
            draws = [statistics.median(rng.choices(samples, k=len(samples))) for _ in range(cfg.bootstrap_draws)]
            estimate = statistics.median(samples)
            result.append({"task": task, "method": method, "case_speedups": values,
                           "speedup": estimate, "slowdown": estimate < 1,
                           "case_bootstrap_95": [percentile(draws, .025), percentile(draws, .975)]})
    return result


def manifest(path):
    write_json(path / "manifest.json", {p.name: sha256(p) for p in sorted(path.iterdir())
                                        if p.is_file() and p.name != "manifest.json"})


def check_manifest(path, expected):
    saved = json.loads((path / "manifest.json").read_text())
    if set(saved) != set(expected) or any(sha256(path / name) != digest for name, digest in saved.items()):
        raise ValueError(f"artifact hash mismatch: {path}")


def run_configuration(path, data, cfg, data_seed, seed, index):
    path = prepare_output(path)
    run_cfg = cfg.run_config(data_seed, seed)
    configure(cfg.threads)
    start = perf_counter_ns()
    model, training = train(data, run_cfg, ModelConfig(width=cfg.width))
    torch.save({"model_config": asdict(model.config), "state_dict": model.state_dict()}, path / "target.pt")
    write_json(path / "config.json", asdict(run_cfg))
    fit_start = perf_counter_ns()
    variants = methods(data, cfg.draft_lengths)
    draft_fit_ns = perf_counter_ns() - fit_start
    print(f"{path.name}: auditing {len(data['eval'])} held-out cases", flush=True)
    references, checks = correctness(model, data, run_cfg, variants)
    for task in TASKS:
        write_gzip(path / f"correctness-{task}.jsonl.gz", [r for r in checks if r["task"] == task])
    timed = timing_cases(data, cfg)
    ids = {r["id"] for r in timed}
    timing_checks = [r for r in checks if r["id"] in ids]
    blocks = []
    order = list(cfg.draft_lengths)[::1 if index % 2 == 0 else -1]
    for k in order:
        block_cfg = cfg.run_config(data_seed, seed, k)
        selected = [v for v in variants if v[0] in ("greedy", f"prompt-k{k}", f"ngram-k{k}")]
        print(f"{path.name}: measuring k={k}, {len(timed)} cases x {cfg.repeats} repeats", flush=True)
        raw = measure(model, {**data, "eval": timed}, block_cfg, selected, references)
        validate_measurements(raw, timing_checks, block_cfg)
        write_gzip(path / f"timings-k{k}.jsonl.gz", raw)
        summary, accuracy = summarize(raw, checks)
        blocks.append({"draft_length": k, "timed_samples": len(raw), "summary": summary,
                       "case_estimates": case_estimates(raw, cfg)})
    report = {"schema_version": 1, "data_seed": data_seed, "training_seed": seed,
              "config": asdict(run_cfg), "model_config": asdict(model.config),
              "dataset_canonical_sha256": object_hash(data), "checkpoint_sha256": sha256(path / "target.pt"),
              "environment": environment(), "training": training, "draft_fit_ns": draft_fit_ns,
              "elapsed_ns": perf_counter_ns() - start, "created_utc": datetime.now(timezone.utc).isoformat(),
              "agreement_passed": True, "evaluation_cases": len(checks), "timing_ids": [r["id"] for r in timed],
              "length_order": order, "measurement_order_seed": seed + 3,
              "blocks": blocks, "accuracy": accuracy}
    write_json(path / "report.json", report)
    manifest(path)
    return report


def verify_configuration(path, data, cfg, data_seed, seed, index):
    expected = {"target.pt", "config.json", "report.json"}
    expected.update(f"correctness-{t}.jsonl.gz" for t in TASKS)
    expected.update(f"timings-k{k}.jsonl.gz" for k in cfg.draft_lengths)
    check_manifest(path, expected)
    run_cfg = cfg.run_config(data_seed, seed)
    if json.loads((path / "config.json").read_text()) != json.loads(json.dumps(asdict(run_cfg))):
        raise AssertionError("configuration differs from protocol")
    configure(cfg.threads)
    saved = torch.load(path / "target.pt", map_location="cpu", weights_only=True)
    if saved["model_config"] != asdict(ModelConfig(width=cfg.width)):
        raise AssertionError("checkpoint dimensions differ from protocol")
    model = TinyTransformer(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    _, replayed = correctness(model, data, run_cfg, methods(data, cfg.draft_lengths))
    checks = [r for task in TASKS for r in read_gzip(path / f"correctness-{task}.jsonl.gz")]
    for old, new in zip(checks, replayed, strict=True):
        for key in new:
            if key == "teacher_forced_nll_sum":
                if not math.isclose(old[key], new[key], rel_tol=1e-6, abs_tol=1e-6):
                    raise AssertionError("replayed loss changed")
            elif old[key] != new[key]:
                raise AssertionError(f"replayed correctness changed: {key}")
    report = json.loads((path / "report.json").read_text())
    order = list(cfg.draft_lengths)[::1 if index % 2 == 0 else -1]
    timed = timing_cases(data, cfg)
    ids = {r["id"] for r in timed}
    timing_checks = [r for r in checks if r["id"] in ids]
    blocks = []
    for k in order:
        raw = read_gzip(path / f"timings-k{k}.jsonl.gz")
        validate_measurements(raw, timing_checks, cfg.run_config(data_seed, seed, k))
        summary, accuracy = summarize(raw, checks)
        blocks.append({"draft_length": k, "timed_samples": len(raw), "summary": summary,
                       "case_estimates": case_estimates(raw, cfg)})
    expected_report = {"schema_version": 1, "data_seed": data_seed, "training_seed": seed,
                       "config": json.loads(json.dumps(asdict(run_cfg))),
                       "model_config": asdict(model.config), "dataset_canonical_sha256": object_hash(data),
                       "checkpoint_sha256": sha256(path / "target.pt"), "agreement_passed": True,
                       "evaluation_cases": len(checks), "timing_ids": [r["id"] for r in timed],
                       "length_order": order, "measurement_order_seed": seed + 3,
                       "blocks": blocks, "accuracy": accuracy}
    for key, value in expected_report.items():
        if report[key] != value:
            raise AssertionError(f"report differs from replay: {key}")
    return report


def collect(reports, cfg):
    cells = []
    for report in reports:
        for block in report["blocks"]:
            for estimate in block["case_estimates"]:
                cells.append({"data_seed": report["data_seed"], "training_seed": report["training_seed"],
                              "draft_length": block["draft_length"], **estimate})
    variation = []
    for task in TASKS:
        for k in cfg.draft_lengths:
            for method in ("greedy", f"prompt-k{k}", f"ngram-k{k}"):
                dataset_estimates = []
                for d in cfg.data_seeds:
                    values = [r["speedup"] for r in cells if r["task"] == task and r["draft_length"] == k
                              and r["method"] == method and r["data_seed"] == d]
                    dataset_estimates.append({"data_seed": d, "training_speedups": values,
                                              "geometric_mean": math.exp(statistics.mean(math.log(v) for v in values))})
                values = [r["geometric_mean"] for r in dataset_estimates]
                variation.append({"task": task, "draft_length": k, "method": method,
                                  "datasets": dataset_estimates, "dataset_min": min(values),
                                  "dataset_median": statistics.median(values), "dataset_max": max(values)})
    return {"agreement_passed": True, "configurations": len(reports),
            "evaluation_cases": sum(r["evaluation_cases"] for r in reports),
            "timed_samples": sum(b["timed_samples"] for r in reports for b in r["blocks"]),
            "cells": cells, "variation": variation,
            "accuracy": [{"data_seed": r["data_seed"], "training_seed": r["training_seed"],
                          "tasks": r["accuracy"]} for r in reports]}


def markdown(report, cfg):
    lines = ["# Dataset replication on CPU", "",
             f"Phase: **{cfg.phase}**. {report['configurations']} targets; {report['evaluation_cases']:,} held-out case/target audits; "
             f"{report['timed_samples']:,} raw timing samples. All audited, warmup and measured outputs matched uncached greedy.", "",
             f"Data seeds {list(cfg.data_seeds)}, paired training seeds {list(cfg.training_seeds)}. Width {cfg.width}, "
             f"two layers, four heads, context 64, vocabulary 22, FP32, {cfg.threads} CPU thread. "
             f"{cfg.steps} training steps, batch {cfg.batch_size}, learning rate {cfg.learning_rate}; "
             f"{cfg.train_per_task} training and {cfg.eval_per_task} held-out examples per task. "
             f"First {cfg.timed_per_task} held-out IDs per task timed; {cfg.warmups} warmups, {cfg.repeats} repetitions. "
             f"Draft lengths {list(cfg.draft_lengths)}; {cfg.max_new_tokens} output-token limit.", "",
             "The protocol was frozen before final training/evaluation. Each length has its own three-method "
             "block and paired greedy baseline; positions are balanced within each case. Configuration order "
             "rotates datasets across training seeds; length order alternates. No result-dependent stopping or selection.", "",
             "## Dataset variation", "",
             "Each dataset column is the geometric mean of its three training-seed estimates. "
             "The range across datasets is descriptive, not a confidence interval. Below 1 means slowdown.", "",
             "| Task | Length | Method | " + " | ".join(str(d) for d in cfg.data_seeds) + " | Dataset min–max |",
             "|---|---:|---|" + "---:|" * len(cfg.data_seeds) + "---|"]
    for row in report["variation"]:
        lines.append(f"| {row['task']} | {row['draft_length']} | {row['method']} | " +
                     " | ".join(f"{d['geometric_mean']:.3f}×" for d in row["datasets"]) +
                     f" | {row['dataset_min']:.3f}–{row['dataset_max']:.3f} |")
    lines += ["", "## Every fitted target", "",
              "Estimate: median across cases of each case's median paired latency ratio. "
              f"Intervals: {cfg.bootstrap_draws} whole-case bootstrap draws, seed {cfg.bootstrap_seed}, percentile 95%. "
              "These intervals are conditional on this model/dataset and do not capture training, dataset or host variation. "
              "Six repetitions stay together; they are not six independent experiments.", "",
              "| Dataset | Train seed | Task | Length | Method | Speedup | Conditional 95% interval | Slowdown |",
              "|---:|---:|---|---:|---|---:|---|---|"]
    for row in report["cells"]:
        if row["method"] == "greedy":
            continue
        lo, hi = row["case_bootstrap_95"]
        lines.append(f"| {row['data_seed']} | {row['training_seed']} | {row['task']} | {row['draft_length']} | "
                     f"{row['method']} | {row['speedup']:.3f}× | {lo:.3f}–{hi:.3f} | {'yes' if row['slowdown'] else 'no'} |")
    lines += ["", "## Target accuracy", "", "Accuracy uses all held-out inputs, separately from decoding equivalence. "
              "Exact continuation includes EOS; missing output tokens count as wrong. Unpredictable continuations "
              "cannot be inferred from the prompt; fast repetitive outputs can still be inaccurate.", "",
              "| Dataset | Train seed | Task | Teacher-forced | Payload only | Free-running | Exact continuation |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for row in report["accuracy"]:
        for task in row["tasks"]:
            lines.append(f"| {row['data_seed']} | {row['training_seed']} | {task['task']} | "
                         f"{task['teacher_forced_token_accuracy']:.2%} | {task['teacher_forced_payload_accuracy']:.2%} | "
                         f"{task['free_running_token_accuracy']:.2%} | {task['exact_continuation_accuracy']:.2%} |")
    lines += ["", "## Artifacts and limits", "",
              "protocol.json freezes settings, dataset hashes, timing IDs, analysis and source hashes. "
              "Each dataset shard stores exact train/evaluation inputs. Each target directory stores its checkpoint, "
              "config, training trace, hardware/dependencies, task correctness shards, raw timing shards, "
              "and report.json with acceptance rates, target calls, draft overhead and all unrounded estimates. "
              "Gzip JSONL shards use deterministic compression and can be read with Python's gzip module.", "",
              ANALYSIS["limits"], "", "See the project README and PROTOCOL.md for reproduction and existing work. "
              "This is replication of established greedy speculative verification and lookup approaches, not a novel algorithm.", ""]
    return "\n".join(lines)


def run_replication(output, protocol):
    frozen, cfg = load_protocol(protocol)
    if frozen["source_sha256"] != environment()["source_sha256"]:
        raise ValueError("implementation changed after freeze; use the recorded source or freeze a new protocol")
    path = prepare_output(output)
    shutil.copyfile(protocol, path / "protocol.json")
    datasets = {}
    for seed in cfg.data_seeds:
        datasets[seed] = generate_data(seed, cfg.train_per_task, cfg.eval_per_task)
        write_gzip(path / f"dataset-{seed}.jsonl.gz", [datasets[seed]])
    reports = []
    for index, (name, data_seed, seed) in enumerate(cfg.configurations()):
        print(f"Running {name} ({index + 1}/{len(cfg.configurations())})", flush=True)
        reports.append(run_configuration(path / name, datasets[data_seed], cfg, data_seed, seed, index))
    report = collect(reports, cfg)
    write_json(path / "replication-report.json", report)
    (path / "RESULTS.md").write_text(markdown(report, cfg))
    files = [p.name for p in path.iterdir() if p.is_file()]
    files += [f"{name}/manifest.json" for name, _, _ in cfg.configurations()]
    write_json(path / "manifest.json", {name: sha256(path / name) for name in sorted(files)})
    return report


def verify_replication(path):
    path = Path(path)
    frozen, cfg = load_protocol(path / "protocol.json")
    files = {"protocol.json", "replication-report.json", "RESULTS.md"}
    files.update(f"dataset-{seed}.jsonl.gz" for seed in cfg.data_seeds)
    files.update(f"{name}/manifest.json" for name, _, _ in cfg.configurations())
    check_manifest(path, files)
    datasets = {}
    for seed in cfg.data_seeds:
        data, = read_gzip(path / f"dataset-{seed}.jsonl.gz")
        if data != generate_data(seed, cfg.train_per_task, cfg.eval_per_task):
            raise AssertionError("dataset does not regenerate")
        audit_splits(data)
        datasets[seed] = data
    reports = []
    for index, (name, data_seed, seed) in enumerate(cfg.configurations()):
        print(f"Replaying {name}", flush=True)
        reports.append(verify_configuration(path / name, datasets[data_seed], cfg, data_seed, seed, index))
    report = collect(reports, cfg)
    if report != json.loads((path / "replication-report.json").read_text()):
        raise AssertionError("aggregate report differs from child results")
    if (path / "RESULTS.md").read_text() != markdown(report, cfg):
        raise AssertionError("markdown differs from recomputed results")
    return {"artifact_hashes_verified": True, "agreement_passed": True, "summary_recomputed": True,
            "splits_disjoint": True, "configurations": len(reports),
            "evaluation_cases": report["evaluation_cases"], "timed_samples": report["timed_samples"],
            "source_hashes_match": frozen["source_sha256"] == environment()["source_sha256"]}
