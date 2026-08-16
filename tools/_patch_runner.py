from pathlib import Path
p = Path(r"tools/run_vgrf_diagnostics.py")
text = p.read_text(encoding="utf-8")
old = """from utils.pressure_alignment import (
    list_reconstructed_segments,
    load_reconstructed_pressure_sensors,
    resolve_reconstructed_rec_dir,
)"""
new = """from utils.pressure_alignment import (
    list_reconstructed_segments,
    load_reconstructed_pressure_sensors,
)"""
if old not in text:
    raise SystemExit("import block not found")
text = text.replace(old, new, 1)

old2 = """def _pair_trials(mocap_root: Path, recon_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for trial_dir in sorted([p for p in mocap_root.iterdir() if p.is_dir()]):
        trial_id = trial_dir.name  # e.g. S13011
        bvh = _find_bvh(trial_dir)
        if bvh is None:
            continue
        # session forms: S1301_1 and S13011
        session_candidates = [trial_id]
        m = re.fullmatch(r\"S(\\d+)(\\d)\", trial_id, flags=re.IGNORECASE)
        if m:
            session_candidates.append(f\"S{m.group(1)}_{m.group(2)}\")
        rec = None
        for sid in session_candidates:
            rec = resolve_reconstructed_rec_dir(sid, recon_root)
            if rec is not None:
                break
        if rec is None:
            continue
        pairs.append((trial_id, bvh, rec))
    return pairs"""

new2 = """def _session_suffixes(trial_id: str) -> list[str]:
    candidates = [trial_id]
    m = re.fullmatch(r\"S(\\d+)(\\d)\", trial_id, flags=re.IGNORECASE)
    if m:
        candidates.append(f\"S{m.group(1)}_{m.group(2)}\")
        candidates.append(f\"S{m.group(1)}{m.group(2)}\")
    m2 = re.fullmatch(r\"S(\\d+)_(\\d+)\", trial_id, flags=re.IGNORECASE)
    if m2:
        candidates.append(f\"S{m2.group(1)}_{m2.group(2)}\")
        candidates.append(f\"S{m2.group(1)}{m2.group(2)}\")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _build_recon_index(recon_root: Path) -> dict[str, Path]:
    \"\"\"Map session suffix (lower) -> reconstructed rec directory.

    Index once to avoid expensive per-trial recursive searches.
    \"\"\"
    index: dict[str, Path] = {}
    for manifest in recon_root.rglob(\"reconstruction_manifest.csv\"):
        rec_dir = manifest.parent
        name = rec_dir.name
        m = re.search(r\"(S\\d+(?:_\\d+)?)$\", name, flags=re.IGNORECASE)
        if not m:
            continue
        suffix = m.group(1)
        key = suffix.lower()
        index.setdefault(key, rec_dir)
        compact = suffix.replace(\"_\", \"\").lower()
        index.setdefault(compact, rec_dir)
    return index


def _pair_trials(mocap_root: Path, recon_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    recon_index = _build_recon_index(recon_root)
    print(f\"recon index size: {len(recon_index)}\")
    for trial_dir in sorted([p for p in mocap_root.iterdir() if p.is_dir()]):
        trial_id = trial_dir.name  # e.g. S13011
        bvh = _find_bvh(trial_dir)
        if bvh is None:
            continue
        rec = None
        for sid in _session_suffixes(trial_id):
            rec = recon_index.get(sid.lower())
            if rec is not None:
                break
        if rec is None:
            continue
        pairs.append((trial_id, bvh, rec))
    return pairs"""

if old2 not in text:
    # show nearby for debug
    idx = text.find("def _pair_trials")
    print(repr(text[idx:idx+600]))
    raise SystemExit("pair function not found")
text = text.replace(old2, new2, 1)
p.write_text(text, encoding="utf-8")
print("updated ok")
