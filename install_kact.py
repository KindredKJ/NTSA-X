"""Install only K-ACT additive files into an existing NTSA-X checkout.

Never copies the bundled source-model reference over existing native_model.py.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import shutil

ADD_FILES=(
    'ntsa_x/kact.py',
    'tests/test_kact.py',
    'benchmark_kact.py',
    'docs/KACT-V0.1.md',
)

def install(source: Path, repo: Path) -> list[str]:
    base=repo/'ntsa_x'/'native_model.py'
    if not base.is_file():
        raise FileNotFoundError(f'No existing NTSA-X native model at {base}')
    actions=[]
    for name in ADD_FILES:
        src=source/name
        dst=repo/name
        if not src.is_file():
            raise FileNotFoundError(f'Missing package component: {src}')
        if dst.exists():
            if sha256(src.read_bytes()).digest()!=sha256(dst.read_bytes()).digest():
                raise FileExistsError(f'Existing different file, refusing to overwrite: {dst}')
            actions.append(f'Unchanged: {name}')
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        actions.append(f'Added: {name}')
    return actions

if __name__=='__main__':
    parser=ArgumentParser(description='Install additive K-ACT prototype files into NTSA-X')
    parser.add_argument('--repository',type=Path,required=True)
    args=parser.parse_args()
    for action in install(Path(__file__).resolve().parent,args.repository.expanduser().resolve()):
        print(action)
