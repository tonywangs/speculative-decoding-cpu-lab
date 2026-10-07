"""Check the size and file-type limits for the project files selected by Git."""
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root
    ).decode().split('\0')
    paths = [root / name for name in set(names) if name]
    total = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'not a regular file: {path.relative_to(root)}')
        size = path.stat().st_size
        if size > 10 * 1024 * 1024:
            raise ValueError(f'file exceeds 10 MiB: {path.relative_to(root)}')
        if path.suffix == '.log' and path.relative_to(root).as_posix() != 'results/tests.log':
            raise ValueError(f'unexpected log: {path.relative_to(root)}')
        total += size
    if len(paths) > 1000 or total > 32 * 1024 * 1024:
        raise ValueError(f'tree exceeds limits: {len(paths)} files, {total} bytes')
    print(f'{len(paths)} project files; {total:,} bytes; all file/tree limits passed.')


if __name__ == '__main__':
    main()
