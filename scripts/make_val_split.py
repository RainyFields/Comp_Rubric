#!/usr/bin/env python
"""Fixed validation split of BrowseComp-Plus (decision 2026-09-24): hold out 100 of the 680 training tasks.

Writes data/bc_val.parquet (100) and data/bc_train_580.parquet (580) and records the split (seed + task ids) in
docs/splits/bc_val_split.json so that any box regenerates the identical files:  python scripts/make_val_split.py [--check]
The 150-task data/bc_test.parquet is untouched (held out until the protocol and checkpoint-selection rules are frozen).
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

SEED, N_VAL = 20260924, 100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, VAL, TRN = (os.path.join(ROOT, "data", f) for f in ("bc_train.parquet", "bc_val.parquet", "bc_train_580.parquet"))
REC = os.path.join(ROOT, "docs", "splits", "bc_val_split.json")

def task_id(row):
    q = row["prompt"][-1]["content"] if len(row["prompt"]) else ""
    return hashlib.sha1((q + "\x1f" + str(row["answer"])).encode()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true", help="verify existing files against the record"); a = ap.parse_args()
    df = pd.read_parquet(SRC); assert len(df) == 680, len(df)
    ids = [task_id(r) for _, r in df.iterrows()]; assert len(set(ids)) == 680, "duplicate task ids"
    order = np.random.RandomState(SEED).permutation(len(df)); val_idx = sorted(order[:N_VAL].tolist()); trn_idx = sorted(order[N_VAL:].tolist())
    rec = {"seed": SEED, "n_val": N_VAL, "source": "data/bc_train.parquet (680 tasks, sha1 of question+answer as task id)",
           "val_task_ids": [ids[i] for i in val_idx], "val_source_indices": val_idx, "val_answers": [str(df.iloc[i]["answer"]) for i in val_idx]}
    if a.check:
        old = json.load(open(REC)); assert old["val_task_ids"] == rec["val_task_ids"], "split record mismatch"
        v = pd.read_parquet(VAL); assert [task_id(r) for _, r in v.iterrows()] == rec["val_task_ids"]; print("split OK:", len(v), "val tasks match the record"); return
    df.iloc[val_idx].reset_index(drop=True).to_parquet(VAL, index=False); df.iloc[trn_idx].reset_index(drop=True).to_parquet(TRN, index=False)
    os.makedirs(os.path.dirname(REC), exist_ok=True); json.dump(rec, open(REC, "w"), indent=1)
    print(f"wrote {VAL} ({len(val_idx)}), {TRN} ({len(trn_idx)}), record {REC}; first ids {rec['val_task_ids'][:3]}")

if __name__ == "__main__": main()
