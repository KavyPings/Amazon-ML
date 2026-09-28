"""Task: (1) snapshot the current output as submissions/v2_3, (2) relearn maps + extract a
closed ~22% training subset (launch/out/big) for the next, larger training run."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if __name__ == "__main__":
    py = sys.executable
    subprocess.call([py, os.path.join(ROOT, "tools", "save_version.py"), "v2_3_dev5states",
                     "--note", "v2.3 model (dev 5 states), val 0.9849, cands 5.2/S1"])
    sys.exit(subprocess.call([py, os.path.join(HERE, "task_extract.py")]))
