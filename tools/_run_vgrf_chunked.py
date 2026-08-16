# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_vgrf_diagnostics import _pair_trials, run_one

def main():
    mocap_root = Path(r"E:/S13/mocap_ori_bvh")
    recon_root = Path(r"E:/S13/reconstruction_20260816_222245")
    out = Path(r"sync/output/vgrf_diagnostics_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    pairs = _pair_trials(mocap_root, recon_root)
    print(f"pairs: {len(pairs)}", flush=True)

    rows = []
    fieldnames = None
    t0 = time.time()
    for idx, (trial_id, bvh, rec) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {trial_id} ...", flush=True)
        try:
            row = run_one(trial_id, bvh, rec)
            row["status"] = "ok"
        except Exception as exc:
            row = {
                "trial_id": trial_id,
                "bvh_path": str(bvh),
                "rec_dir": str(rec),
                "status": "error",
                "error": str(exc),
            }
            print(f"  ERROR {exc}", flush=True)
        rows.append(row)

        # rewrite CSV each row so partial results are available
        if fieldnames is None:
            fieldnames = list(row.keys())
        else:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote partial rows={len(rows)} elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"DONE wrote {out} rows={len(rows)} elapsed={time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
