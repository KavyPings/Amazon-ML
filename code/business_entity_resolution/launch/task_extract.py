"""Relearn normalisation maps on the FULL training set (current transliteration) and extract a
CLOSED training subset (whole states) as small gzip parts for cloud-side training."""
import gzip, os, pickle, random, sys, time, zlib
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.config import Config
from ber.io import find_data_dir, load_split, gt_to_dict, log
from ber.normalize import learn_maps
from ber.pipeline import closed_subset

FRAC = float(os.environ.get("BER_FRAC", "0.22"))
SEED = int(os.environ.get("BER_SEED", "7"))
if __name__ == "__main__":
    T0 = time.time()
    OUT = os.path.join(os.path.dirname(__file__), "out", "big")
    os.makedirs(OUT, exist_ok=True)
    data = load_split(find_data_dir(None), "train")
    truth = gt_to_dict(data["gt"])
    pool = pd.concat([data["s2"], data["s3"]], ignore_index=True)
    maps = learn_maps(data["s1"], pool, truth, Config(), random.Random(0))
    with open(os.path.join(OUT, "maps_full2.pkl"), "wb") as f:
        pickle.dump(maps, f)
    log(f"maps {({k: len(v) for k, v in maps.items()})}")
    s1, pl = closed_subset(data["s1"], pool, truth, maps, FRAC, SEED)
    gt = data["gt"][data["gt"]["source1_entity_id"].isin(set(s1["entity_id"]))]

    def dump(df, name, rows_per_part=400_000):
        for k, lo in enumerate(range(0, len(df), rows_per_part)):
            with gzip.open(os.path.join(OUT, f"{name}.part{k}.tsv.gz"), "wt", encoding="utf-8", newline="") as f:
                df.iloc[lo:lo + rows_per_part].to_csv(f, sep="\t", index=False)
    cols = ["entity_id", "business_name", "business_address", "country"]
    dump(s1[cols], "train_source1")
    dump(pl[pl["entity_id"].str.startswith("S2-")][cols], "train_source2")
    dump(pl[pl["entity_id"].str.startswith("S3-")][cols], "train_source3")
    dump(gt, "train_ground_truth")
    log(f"subset S1={len(s1):,} pool={len(pl):,} done in {time.time()-T0:.0f}s")
