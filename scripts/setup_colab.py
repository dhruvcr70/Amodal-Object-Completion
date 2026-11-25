"""Create an isolated baseline environment in a Colab Linux runtime."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path('/content/amodal-completion')
ROOT.mkdir(exist_ok=True)

def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)

run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
uv = [sys.executable, '-m', 'uv']
install_env = os.environ.copy()
# Colab's CPU and CUDA package indexes expose different wheels. uv's default
# first-index policy can incorrectly conclude torchmetrics is missing from the
# CUDA index even when it is available on PyPI.
install_env['UV_INDEX_STRATEGY'] = 'unsafe-best-match'
env = ROOT / '.venv'
run(uv + ['venv', '--python', '3.10', str(env)])
python = env / 'bin/python'
run(uv + ['pip', 'install', '--python', str(python), '-r',
          str(ROOT/'requirements-baseline.txt'),
          '--extra-index-url', 'https://download.pytorch.org/whl/cu118'],
    env=install_env)
# PyTorch Lightning 1.9 still imports pkg_resources, which setuptools removed
# from recent releases. Pin the compatibility version explicitly.
run(uv + ['pip', 'install', '--python', str(python), 'setuptools<81',
          '--index-url', 'https://pypi.org/simple'])

third_party = ROOT / 'third_party'
third_party.mkdir(exist_ok=True)
repositories = {
    'pix2gestalt': 'https://github.com/cvlab-columbia/pix2gestalt.git',
    'taming-transformers': 'https://github.com/CompVis/taming-transformers.git',
    'CLIP': 'https://github.com/openai/CLIP.git',
}
revisions = {}
for name, url in repositories.items():
    dest = third_party/name
    if not dest.exists():
        run(['git', 'clone', '--depth', '1', url, str(dest)])
    revisions[name] = subprocess.check_output(
        ['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if name != 'pix2gestalt':
        run(uv + ['pip','install','--python',str(python),'--no-deps','-e',str(dest)])

# The old Lightning import moved; make the compatibility edit explicit and recorded.
ddpm = third_party/'pix2gestalt/pix2gestalt/ldm/models/diffusion/ddpm.py'
source = ddpm.read_text()
old = 'from pytorch_lightning.utilities.distributed import rank_zero_only'
new = 'from pytorch_lightning.utilities.rank_zero import rank_zero_only'
ddpm.write_text(source.replace(old, new))
manifest = {'repositories': revisions,
            'patches': [f'{old} -> {new}', 'setuptools<81 for PyTorch Lightning pkg_resources compatibility'],
            'python': str(python)}
(ROOT/'environment_manifest.json').write_text(json.dumps(manifest, indent=2))
check = subprocess.run([str(python), '-c',
    "import torch, pytorch_lightning; print('Isolated PyTorch:', torch.__version__); print('Lightning:', pytorch_lightning.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Select a T4 GPU runtime before continuing.'"],
    check=True, text=True, capture_output=True)
print(check.stdout, end='')
print('Environment created. Full model inference is the next separate check.')
