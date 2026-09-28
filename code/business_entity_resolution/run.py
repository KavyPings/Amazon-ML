"""Convenience launcher so nobody needs to set PYTHONPATH.

    python run.py all                 # train + predict + validate
    python run.py train --set train_sample_s1=0.2   # quick dry run on 20% of train
    python run.py predict
"""
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows-safe
    from ber.__main__ import main
    sys.exit(main())
