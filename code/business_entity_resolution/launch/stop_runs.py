"""Stop any running pipeline task started from launch\\go*.bat (task*.py, and run.py steps with
their worker processes)."""
import os
import re

import psutil

me = os.getpid()
pat = re.compile(r"(task\w*\.py|run\.py)")
killed = []
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        cmd = " ".join(p.info["cmdline"] or [])
        if p.info["pid"] != me and "python" in (p.info["name"] or "").lower() and pat.search(cmd):
            for ch in p.children(recursive=True):
                ch.kill()
            p.kill()
            killed.append(p.info["pid"])
    except Exception:
        pass
with open(os.path.join(os.path.dirname(__file__), "stop_log.txt"), "w") as f:
    f.write(f"killed {killed}\n")
