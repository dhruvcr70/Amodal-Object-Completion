"""Report hardware without requiring ML packages or downloading model weights."""
from pathlib import Path
import importlib.util
import json
import platform
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
report={'python':sys.version,'interpreter':sys.executable,'platform':platform.platform()}
try:
    report['nvidia_smi']=subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
        text=True,timeout=15).strip()
except (OSError,subprocess.SubprocessError) as error:
    report['nvidia_smi_error']=str(error)
if importlib.util.find_spec('torch'):
    try:
        import torch
        report.update(torch_version=torch.__version__,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report['torch_gpu']=torch.cuda.get_device_name(0)
    except Exception as error:
        report['torch_error']=str(error)
else:
    report['pytorch_status']='Not installed in this project environment yet.'
dest=root/'runs'/'local-hardware.json'
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print('\nSaved report:',dest)
