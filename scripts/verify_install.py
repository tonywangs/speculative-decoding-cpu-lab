"""Build a wheel, install it in a clean venv, and exercise it outside the source tree.

Run with a Python that has pip >= 22.3. Downloads public CPU wheels only. The
isolated environment and smoke artifacts are temporary; captured results live
separately in results/. No packages are installed into the invoking environment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def checked(command, cwd):
    print(json.dumps([str(x) for x in command]), flush=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run([str(x) for x in command], cwd=cwd, check=True, env=env)


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="specdecode-install-") as temp:
        work = Path(temp)
        wheel_dir = work / "wheel"
        checked([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps",
                 "--wheel-dir", wheel_dir, root], work)
        target = work / "venv"
        venv.EnvBuilder(with_pip=False).create(target)
        python = target / "bin" / "python"
        cli = target / "bin" / "specdecode"
        wheel = next(wheel_dir.glob("specdecode_cpu_lab-*.whl"))
        checked([sys.executable, "-m", "pip", "--python", python, "install", "--no-cache-dir",
                 "-r", root / "requirements-cpu.txt", wheel], work)
        checked([python, "-c", "import specdecode; print(specdecode.__file__)"], work)
        checked([python, "-m", "unittest", "discover", "-s", root / "tests", "-v"], work)
        output = work / "experiment"
        checked([cli, "run", "--output", output, "--steps", "8", "--train-per-task", "12",
                 "--eval-per-task", "2", "--warmups", "1", "--repeats", "2",
                 "--draft-lengths", "1", "2", "4", "8"], work)
        checked([cli, "verify", output], work)
        checked([cli, "benchmark", output, "--output", work / "retimed"], work)
        checked([cli, "verify", work / "retimed"], work)
        suite = work / "suite"
        checked([cli, "suite", "--output", suite, "--steps", "2", "--train-per-task", "4",
                 "--eval-per-task", "1", "--warmups", "1", "--repeats", "5",
                 "--draft-lengths", "1", "2", "--widths", "8", "16", "--max-new-tokens", "4"], work)
        checked([cli, "verify-suite", suite], work)
        checked([cli, "verify-suite", root / "results" / "cpu-multiseed"], work)
        print("Isolated wheel installation, tests, run, verify, benchmark and suite replay passed.")


if __name__ == "__main__":
    main()
