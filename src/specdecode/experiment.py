"""Training, paired measurements and self-contained experiment artifacts."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
from time import perf_counter_ns

import torch
import torch.nn.functional as F

from .data import EOS, TASKS, generate_data
from .decode import decode, uncached_greedy
from .draft import NGram, PromptLookup
from .model import ModelConfig, TinyTransformer


@dataclass(frozen=True)
class RunConfig:
    seed: int = 17
    train_per_task: int = 256
    eval_per_task: int = 12
    steps: int = 800
    batch_size: int = 32
    learning_rate: float = 0.003
    threads: int = 1
    warmups: int = 2
    repeats: int = 5
    draft_lengths: tuple[int, ...] = (1, 2, 4, 8)
    max_new_tokens: int = 25

    def __post_init__(self):
        for key in ("train_per_task", "eval_per_task", "steps", "batch_size", "threads", "repeats", "max_new_tokens"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.warmups < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("warmups and learning_rate must be positive")
        if not self.draft_lengths or any(k < 1 for k in self.draft_lengths) or len(set(self.draft_lengths)) != len(self.draft_lengths):
            raise ValueError("draft_lengths must be unique positive integers")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure(threads):
    torch.set_num_threads(threads)
    # Set once before parallel work; library callers may already have set this.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def environment():
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((s.split(":", 1)[1].strip() for s in cpuinfo.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_model": cpu, "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch": torch.__version__, "torch_build": torch.__config__.show(),
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "dtype": "float32", "device": "cpu", "attention": "explicit masked matmul + softmax",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def train(data, cfg, model_cfg):
    total_start = perf_counter_ns()
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    sampling = torch.Generator().manual_seed(cfg.seed + 2)
    tokens = torch.tensor([row["tokens"] for row in data["train"]], dtype=torch.long)
    log = []
    start = perf_counter_ns()
    for step in range(cfg.steps):
        ids = torch.randint(len(tokens), (cfg.batch_size,), generator=sampling)
        batch = tokens[ids]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, model_cfg.vocab_size), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            log.append({"step": step + 1, "batch_loss": float(loss.detach()),
                        "elapsed_ns": perf_counter_ns() - start})
    elapsed = perf_counter_ns() - start
    model.eval()
    return model, {"elapsed_ns": elapsed, "setup_and_training_ns": perf_counter_ns() - total_start,
                   "trace": log, "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                   "batch_sampling": "uniform with replacement", "scheduler": None,
                   "weight_decay": 0.01, "gradient_clip_norm": 1.0,
                   "initialization_seed": cfg.seed, "batch_sampling_seed": cfg.seed + 2,
                   "parameter_count": sum(p.numel() for p in model.parameters())}


def methods(data, lengths):
    ngram = NGram([r["tokens"] for r in data["train"]])
    lookup = PromptLookup()
    return [("greedy", None, 0)] + [(f"{name}-k{k}", draft, k)
        for name, draft in (("prompt", lookup), ("ngram", ngram)) for k in lengths]


def assert_agreement(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"greedy disagreement: {label}: {actual} != {expected}")


@torch.inference_mode()
def correctness(model, data, cfg, variants):
    rows = []
    references = {}
    for case in data["eval"]:
        p = case["prompt_length"]
        prompt, truth = case["tokens"][:p], case["tokens"][p:]
        ref = uncached_greedy(model, prompt, cfg.max_new_tokens)
        references[case["id"]] = ref
        counts = {}
        for name, drafter, k in variants:
            got = decode(model, prompt, cfg.max_new_tokens, drafter, k, audit=True)
            assert_agreement(got.tokens, ref, f"{case['id']}/{name}")
            counts[name] = {key: value for key, value in asdict(got).items() if key != "draft_ns"}
        logits, _ = model(torch.tensor([case["tokens"][:-1]]))
        pred = logits[0, p - 1:].argmax(-1).tolist()
        nll = float(F.cross_entropy(logits[0, p - 1:], torch.tensor(truth), reduction="sum"))
        rows.append({"id": case["id"], "task": case["task"], "expected_tokens": truth,
                     "greedy_tokens": ref, "teacher_forced_correct": sum(a == b for a, b in zip(pred, truth)),
                     "teacher_forced_payload_correct": sum(a == b and b != EOS for a, b in zip(pred, truth)),
                     "continuation_tokens": len(truth), "payload_tokens": sum(t != EOS for t in truth),
                     "teacher_forced_nll_sum": nll,
                     "free_running_correct": sum(a == b for a, b in zip(ref, truth)),
                     "exact_continuation": ref == truth, "methods": counts})
    return references, rows


def measure(model, data, cfg, variants, references):
    rng = random.Random(cfg.seed + 3)
    cases = list(data["eval"])
    # Warm all methods and all prompts outside measurements.
    for _ in range(cfg.warmups):
        for case in cases:
            for name, drafter, k in variants:
                got = decode(model, case["tokens"][:case["prompt_length"]], cfg.max_new_tokens, drafter, k)
                assert_agreement(got.tokens, references[case["id"]], f"warmup/{case['id']}/{name}")
    raw = []
    for repeat in range(cfg.repeats):
        rng.shuffle(cases)
        for case in cases:
            order = list(variants)
            rng.shuffle(order)
            for position, (name, drafter, k) in enumerate(order):
                prompt = case["tokens"][:case["prompt_length"]]
                start = perf_counter_ns()
                got = decode(model, prompt, cfg.max_new_tokens, drafter, k)
                elapsed = perf_counter_ns() - start
                assert_agreement(got.tokens, references[case["id"]], f"timing/{case['id']}/{name}")
                metrics = asdict(got)
                metrics.pop("tokens")
                raw.append({"id": case["id"], "task": case["task"], "repeat": repeat,
                            "order_in_pair": position, "method": name, "elapsed_ns": elapsed,
                            "generated_tokens": len(got.tokens), **metrics})
    return raw


def summarize(raw, correctness_rows):
    summary = []
    baseline = {(r["id"], r["repeat"]): r["elapsed_ns"] for r in raw if r["method"] == "greedy"}
    def method_key(name):
        if name == "greedy":
            return (0, 0)
        family, length = name.split("-k")
        return (1 if family == "prompt" else 2, int(length))

    for task in TASKS:
        for method in sorted({r["method"] for r in raw}, key=method_key):
            rows = [r for r in raw if r["task"] == task and r["method"] == method]
            ratios = [baseline[r["id"], r["repeat"]] / r["elapsed_ns"] for r in rows]
            proposed = sum(r["proposed"] for r in rows)
            summary.append({"task": task, "method": method, "samples": len(rows),
                "median_latency_ms": statistics.median(r["elapsed_ns"] for r in rows) / 1e6,
                "paired_speedup_median": statistics.median(ratios),
                "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios),
                "aggregate_speedup": sum(baseline[r["id"], r["repeat"]] for r in rows) / sum(r["elapsed_ns"] for r in rows),
                "acceptance_rate": sum(r["accepted"] for r in rows) / proposed if proposed else None,
                "mean_forward_calls": statistics.mean(r["forward_calls"] for r in rows),
                "mean_forward_tokens": statistics.mean(r["forward_tokens"] for r in rows),
                "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in rows),
                "median_draft_ms": statistics.median(r["draft_ns"] for r in rows) / 1e6})
    accuracy = []
    for task in TASKS:
        rows = [r for r in correctness_rows if r["task"] == task]
        n = sum(r["continuation_tokens"] for r in rows)
        accuracy.append({"task": task, "cases": len(rows),
            "teacher_forced_token_accuracy": sum(r["teacher_forced_correct"] for r in rows) / n,
            "teacher_forced_payload_accuracy": sum(r["teacher_forced_payload_correct"] for r in rows) / sum(r["payload_tokens"] for r in rows),
            "teacher_forced_nll": sum(r["teacher_forced_nll_sum"] for r in rows) / n,
            "free_running_token_accuracy": sum(r["free_running_correct"] for r in rows) / n,
            "exact_continuation_accuracy": sum(r["exact_continuation"] for r in rows) / len(rows)})
    return summary, accuracy


def report_markdown(report):
    lines = ["# Measured CPU results", "", f"Recorded at {report['created_utc']}.", "",
             f"CPU: {report['environment']['cpu_model']}; PyTorch {report['environment']['torch']}; "
             f"{report['environment']['intraop_threads']} intra-op thread(s), FP32.", "",
             "Speedup is the median of greedy/method latency ratios paired by case and repetition. "
             "Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; "
             "training, n-gram fitting, correctness audits and warmups are excluded.", "",
             "| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        acc = "—" if row["acceptance_rate"] is None else f"{row['acceptance_rate']:.1%}"
        lines.append(f"| {row['task']} | {row['method']} | {row['median_latency_ms']:.3f} | "
                     f"{row['paired_speedup_median']:.3f}× | {acc} | {row['mean_forward_calls']:.2f} | {row['median_draft_ms']:.3f} |")
    lines.extend(["", "| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |",
                  "|---|---:|---:|---:|"])
    for row in report["accuracy"]:
        lines.append(f"| {row['task']} | {row['teacher_forced_token_accuracy']:.1%} | "
                     f"{row['free_running_token_accuracy']:.1%} | {row['exact_continuation_accuracy']:.1%} |")
    lines.extend(["", "Accuracy is against held-out synthetic continuations, including EOS. "
                  "See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.", "",
                  "Every measured output matched the uncached greedy reference. This is empirical token agreement, "
                  "not bitwise equality of floating-point logits or a guarantee on other hardware.", "",
                  "These are small synthetic workloads and one trained target. Random task continuations are "
                  "not predictable from the prompt. Target repetition can make them easy to draft despite low task "
                  "accuracy. Repeated timing samples share prompts and weights and are not independent model trials. "
                  "Shared-host load can affect latency. No claims about natural language or large-model speedups.", ""])
    return "\n".join(lines)


def finish_report(output, model, data, cfg, training, checkpoint_hash):
    start = perf_counter_ns()
    variants = methods(data, cfg.draft_lengths)
    draft_fit_ns = perf_counter_ns() - start
    print("Auditing cache rollback and greedy agreement...", flush=True)
    references, checks = correctness(model, data, cfg, variants)
    print("Running warmups and paired timings...", flush=True)
    raw = measure(model, data, cfg, variants, references)
    summary, accuracy = summarize(raw, checks)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": asdict(cfg), "model_config": asdict(model.config), "environment": environment(),
              "training": training, "draft_fit_ns": draft_fit_ns, "checkpoint_sha256": checkpoint_hash,
              "measurement_order_seed": cfg.seed + 3, "agreement_passed": True,
              "evaluation_cases": len(checks), "timed_samples": len(raw),
              "summary": summary, "accuracy": accuracy}
    write_json(output / "correctness.json", checks)
    with (output / "timings.jsonl").open("w") as stream:
        for row in raw:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(output / "report.json", report)
    (output / "RESULTS.md").write_text(report_markdown(report))
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"})
    return report


def run(output, cfg):
    output = prepare_output(output)
    configure(cfg.threads)
    data = generate_data(cfg.seed, cfg.train_per_task, cfg.eval_per_task)
    write_json(output / "dataset.json", data)
    print(f"Training {cfg.steps} steps on {len(data['train'])} synthetic sequences...", flush=True)
    model, training = train(data, cfg, ModelConfig())
    torch.save({"model_config": asdict(model.config), "state_dict": model.state_dict()}, output / "target.pt")
    write_json(output / "config.json", asdict(cfg))
    return finish_report(output, model, data, cfg, training, sha256(output / "target.pt"))


def load_experiment(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    required = {"config.json", "dataset.json", "target.pt", "report.json", "correctness.json", "timings.jsonl", "RESULTS.md"}
    if not required.issubset(manifest):
        raise ValueError("incomplete artifact manifest")
    for name, digest in manifest.items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    cfg = RunConfig(**json.loads((path / "config.json").read_text()))
    configure(cfg.threads)
    data = json.loads((path / "dataset.json").read_text())
    if data != generate_data(cfg.seed, cfg.train_per_task, cfg.eval_per_task):
        raise ValueError("dataset does not match generator/config")
    saved = torch.load(path / "target.pt", map_location="cpu", weights_only=True)
    model = TinyTransformer(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, data, cfg


def verify(path):
    model, data, cfg = load_experiment(path)
    references, rows = correctness(model, data, cfg, methods(data, cfg.draft_lengths))
    recorded = json.loads((Path(path) / "correctness.json").read_text())
    for old, new in zip(recorded, rows, strict=True):
        if old["id"] != new["id"]:
            raise AssertionError("case ordering changed")
        assert_agreement(old["greedy_tokens"], new["greedy_tokens"], old["id"])
        if old["methods"] != new["methods"]:
            raise AssertionError("decoding counters changed")
    report = json.loads((Path(path) / "report.json").read_text())
    raw = [json.loads(line) for line in (Path(path) / "timings.jsonl").read_text().splitlines()]
    summary, accuracy = summarize(raw, recorded)
    if summary != report["summary"] or accuracy != report["accuracy"]:
        raise AssertionError("recorded summary does not match raw measurements")
    if len(raw) != len(data["eval"]) * cfg.repeats * (1 + 2 * len(cfg.draft_lengths)):
        raise AssertionError("incomplete paired measurements")
    source_matches = report["environment"]["source_sha256"] == environment()["source_sha256"]
    return {"artifact_hashes_verified": True, "agreement_passed": True,
            "summary_recomputed": True, "source_hashes_match": source_matches, "cases": len(references)}
