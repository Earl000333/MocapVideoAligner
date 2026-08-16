import os, sys, runpy
from pathlib import Path
os.environ["PYTHONPATH"] = str(Path.cwd())
sys.argv = [
    "tools/run_vgrf_diagnostics.py",
    "--mocap-root", r"E:\S13\mocap_ori_bvh",
    "--recon-root", r"E:\S13\reconstruction_20260816_222245",
    "--output", r"sync\output\vgrf_diagnostics_report.csv",
]
runpy.run_path("tools/run_vgrf_diagnostics.py", run_name="__main__")
