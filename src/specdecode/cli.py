import argparse
import json
from pathlib import Path

from .suite import SuiteConfig, run_suite, verify_suite

from .experiment import RunConfig, finish_report, load_experiment, prepare_output, run, sha256, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic CPU speculative decoding experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="generate, train, audit, benchmark and save artifacts")
    run_parser.add_argument("--output", type=Path, required=True)
    for name in ("seed", "train_per_task", "eval_per_task", "steps", "batch_size", "threads", "warmups", "repeats", "max_new_tokens"):
        run_parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(RunConfig(), name))
    run_parser.add_argument("--learning-rate", type=float, default=RunConfig().learning_rate)
    run_parser.add_argument("--draft-lengths", type=int, nargs="+", default=list(RunConfig().draft_lengths))
    run_parser.add_argument("--data-seed", type=int)
    run_parser.add_argument("--width", type=int, default=64)
    run_parser.add_argument("--balanced-order", action="store_true")
    suite_parser = sub.add_parser("suite", help="train and compare three seeds and two sizes on identical data")
    suite_parser.add_argument("--output", type=Path, required=True)
    for name in ("data_seed", "train_per_task", "eval_per_task", "steps", "batch_size", "threads", "warmups", "repeats", "max_new_tokens"):
        suite_parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(SuiteConfig(), name))
    suite_parser.add_argument("--learning-rate", type=float, default=SuiteConfig().learning_rate)
    for name in ("seeds", "widths", "draft_lengths"):
        suite_parser.add_argument("--" + name.replace("_", "-"), type=int, nargs="+", default=list(getattr(SuiteConfig(), name)))
    suite_verify = sub.add_parser("verify-suite", help="verify every suite artifact and replay all six targets")
    suite_verify.add_argument("experiment", type=Path)
    verify_parser = sub.add_parser("verify", help="verify hashes and replay correctness without training/timing")
    verify_parser.add_argument("experiment", type=Path)
    bench_parser = sub.add_parser("benchmark", help="repeat timings using an existing saved model and data")
    bench_parser.add_argument("experiment", type=Path)
    bench_parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "verify-suite":
            print(json.dumps(verify_suite(args["experiment"]), indent=2))
            return
        if command == "suite":
            output = args.pop("output")
            report = run_suite(output, SuiteConfig(**args))
        elif command == "run":
            output = args.pop("output")
            report = run(output, RunConfig(**args))
        elif command == "verify":
            print(json.dumps(verify(args["experiment"]), indent=2))
            return
        else:
            import shutil
            source = args["experiment"]
            model, data, cfg = load_experiment(source)
            output = prepare_output(args["output"])
            for name in ("target.pt", "dataset.json", "config.json"):
                shutil.copyfile(source / name, output / name)
            old = json.loads((source / "report.json").read_text())
            report = finish_report(output, model, data, cfg, old["training"], sha256(output / "target.pt"))
        print(json.dumps({"output": str(output), "agreement_passed": report["agreement_passed"],
                          "evaluation_cases": report["evaluation_cases"], "timed_samples": report["timed_samples"]}, indent=2))
    except (ValueError, AssertionError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
