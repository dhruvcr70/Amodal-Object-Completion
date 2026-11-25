"""Start the local SAM-enabled Gradio mask demo as a detached Windows process."""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 7860
URL = f"http://{HOST}:{PORT}/"
LOG_DIR = ROOT / "runs" / "gradio-local"
LOG_DIR.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    str(ROOT / "scripts" / "demo_pix2gestalt.py"),
    "--host", HOST,
    "--port", str(PORT),
    "--sam-checkpoint", str(ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"),
]

with (LOG_DIR / "server.log").open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=(subprocess.DETACHED_PROCESS
                       | subprocess.CREATE_NEW_PROCESS_GROUP
                       | subprocess.CREATE_NO_WINDOW),
        close_fds=True,
    )

for _ in range(30):
    try:
        with urllib.request.urlopen(URL, timeout=1) as response:
            if response.status == 200:
                print(f"Local demo is running at {URL} (PID {process.pid}).")
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)

print(f"Demo did not start. Check {LOG_DIR / 'server.log'}")
raise SystemExit(1)
