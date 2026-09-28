"""Central configuration for the Business Entity Resolution pipeline.

Every tunable number lives here so teammates can change behaviour from the
command line (``--set key=value``) without editing code.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # ---------------- general ----------------
    seed: int = 42
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)   # threads for LightGBM / top-k search
    val_fraction: float = 0.2          # share of train S1 entities held out as a mock test set
    train_state_frac: float = 1.0      # <1.0 = train on a CLOSED subset: this share of states (all their records)
    maps_path: str = ""                # optional pre-learned normalisation maps (pickle)

    # ---------------- normalisation ----------------
    abbrev_min_count: int = 20         # an abbreviation pair must be seen this often in true matches
    abbrev_min_share: float = 0.5      # ...and be the dominant expansion of that short form

    # ---------------- blocking (candidate generation) ----------------
    knn_k_per_source: int = 10         # nearest names kept per S1 per source (S2, S3)
    knn_min_sim: float = 0.20          # cosine floor on char-3gram TF-IDF of the core name
    comb_w_name: float = 0.8           # weight of the name part in the combined ranking vector
    comb_w_addr: float = 0.6           # weight of the address part
    addr_k_per_source: int = 5         # name neighbours kept inside a postcode partition
    addr_max_block: int = 30           # exact address-key blocks larger than this are skipped
    knn_max_partition: int = 3000      # partitions with more candidates than this get split further
    gen_addr_heavy: bool = True        # extra pass: name partitions ranked by an ADDRESS-heavy score
    ah_scale_name: float = 0.5         # query-side factor on the name block for that pass
    ah_scale_addr: float = 2.0         # query-side factor on the address block for that pass
    ah_k_per_source: int = 5
    gen_numgram: bool = True           # key join on consecutive address-number pairs ("16 11", "4024 39")
    numgram_max_block: int = 20
    gen_reverse: str = "off"           # pool->S1 name top-k for pool records: "empty" (no address), "all", "off"
    rev_k: int = 3
    matcher_train_frac: float = 1.0    # experiments only: share of training S1 used to fit the matcher
    pruner_dir: str = ""               # two-phase training: reuse pruner.txt + params.json from this folder
    save_all_scores: bool = False      # debug: also write every scored training pair (all_scores.parquet)

    # ---------------- pruning (learned blocking, stage 1) ----------------
    s1_chunk: int = 1_000_000             # S1 records processed per blocking chunk (memory bound)
    cap_by_best: bool = True           # cap raw pairs by max(name-led, address-led cosine) + generator votes
    raw_cap_per_src: int = 20           # raw pairs kept per (S1, source) by combined cosine (memory bound)
    prune_target_recall: float = 0.995  # keep this share of the true pairs that blocking found (0.998: +10-15% candidates, +0.0001..0.0007 F0.5)
    prune_min_p: float = 0.01           # never keep candidates below this pruner probability
    max_candidates: int = 12            # hard cap on candidates per S1 after pruning (max true matches is 11)

    # ---------------- matcher (stage 2) ----------------
    lgb_num_leaves: int = 127
    lgb_learning_rate: float = 0.05
    lgb_rounds: int = 3000
    lgb_early_stop: int = 50
    lgb_min_child_samples: int = 100
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8

    # ---------------- decoding ----------------
    # filled in automatically by `train` (grid search on the validation split)
    decode_method: str = "efdecode"    # "efdecode" or "threshold"
    decode_alpha: float = 1.0          # weight on the "predict nothing" option
    decode_threshold: float = 0.5
    decode_min_p: float = 0.05         # candidates below this prob are never predicted
    one_owner: bool = True             # each S2/S3 record may belong to only one S1
    one_owner_margin: float = 0.0      # top owner must beat the runner-up by this much

    # ------------------------------------------------------------------
    def update(self, overrides: dict) -> "Config":
        """Apply ``{name: value}`` overrides, casting strings to the field type."""
        types = {f.name: f.type for f in fields(self)}
        for k, v in overrides.items():
            if k not in types:
                raise KeyError(f"Unknown config key: {k}")
            cur = getattr(self, k)
            if isinstance(v, str):
                if isinstance(cur, bool):
                    v = v.lower() in ("1", "true", "yes", "y")
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
            setattr(self, k, v)
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        known = {f.name for f in fields(cfg)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
        return cfg
