# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_vgrf_diagnostics import _pair_trials, run_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', type=Path, default=Path('sync/output/vgrf_diagnostics_report.csv'))
    ap.add_argument('--append', action='store_true')
    args = ap.parse_args()

    pairs = _pair_trials(Path(r'E:/S13/mocap_ori_bvh'), Path(r'E:/S13/reconstruction_20260816_222245'))
    if args.offset:
        pairs = pairs[args.offset:]
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f'batch pairs={len(pairs)} offset={args.offset}', flush=True)

    rows = []
    if args.append and args.output.exists():
        with args.output.open('r', encoding='utf-8-sig', newline='') as f:
            r = csv.DictReader(f)
            rows = list(r)
            fieldnames = list(r.fieldnames or [])
    else:
        fieldnames = []

    t0 = time.time()
    for idx, (trial_id, bvh, rec) in enumerate(pairs, 1):
        print(f'[{idx}/{len(pairs)}] {trial_id}', flush=True)
        try:
            row = run_one(trial_id, bvh, rec)
            row['status'] = 'ok'
        except Exception as exc:
            row = {'trial_id': trial_id, 'bvh_path': str(bvh), 'rec_dir': str(rec), 'status': 'error', 'error': str(exc)}
            print(' ERROR', exc, flush=True)
        rows.append(row)
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
    print(f'DONE batch rows_total={len(rows)} elapsed={time.time()-t0:.1f}s', flush=True)

if __name__ == '__main__':
    main()
