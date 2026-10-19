"""Test the installed real-text CLI outside the checkout, using existing Torch.

Build isolation downloads public build dependencies. The temporary wheel is
installed without runtime dependencies into a separate import directory; the
invoking Python's already installed Torch is used. This is package isolation,
not a claim to test a fresh runtime dependency installation. Historical synthetic
archives are not replayed because the existing unit suite covers shared changes.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='specdecode-realtext-wheel-') as temp:
        work = Path(temp)
        wheels = work/'wheels'
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-cache-dir',
                        '--wheel-dir', str(wheels), str(root)], cwd=work, check=True)
        wheel = next(wheels.glob('specdecode_cpu_lab-*.whl'))
        package = work/'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                        '--target', str(package), str(wheel)], cwd=work, check=True)
        env = {**os.environ, 'PYTHONPATH': str(package), 'PYTHONHASHSEED': '71'}
        env.pop('PYTHONHOME', None)
        def checked(args):
            print('Checking:', args, flush=True)
            subprocess.run([sys.executable, *map(str, args)], cwd=work, env=env, check=True)
        checked(['-c', 'import pathlib, specdecode, sys; '
                 'assert pathlib.Path(specdecode.__file__).is_relative_to(pathlib.Path(sys.argv[1])); '
                 'print(specdecode.__file__)', package])
        checked(['-m', 'unittest', 'discover', '-s', root/'tests', '-p', 'test_recurrent.py', '-v'])
        corpus = root/'results/realtext-gru/corpus'
        for seed in (17, 29):
            recorded = root/f'results/realtext-gru/seed{seed}'
            checked(['-m', 'specdecode.cli', 'verify-realtext', recorded, '--corpus', corpus])
        for index in (0, 12, 24):
            checked(['-m', 'specdecode.cli', 'demo-realtext', root/'results/realtext-gru/seed17',
                     '--corpus', corpus, '--case-index', index])
        print('Installed wheel: GRU tests, both recorded experiments and three live genre demos passed.')


if __name__ == '__main__':
    main()
