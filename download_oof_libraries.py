"""
download_oof_libraries.py
=========================
Downloads all public OOF prediction libraries for S6E8 Smartphone Addiction
competition. These libraries contain pre-computed out-of-fold (OOF) and test
prediction arrays aligned to the frozen 5-fold split:

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    over train.csv in ORIGINAL FILE ROW ORDER — never sorted, never reindexed.

Usage:
    python download_oof_libraries.py

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
      (Download from https://www.kaggle.com/settings -> API -> Create New Token)
    - `kaggle` CLI installed: pip install kaggle

Downloads into: input/oof_libraries/<dataset-slug>/

Leave-one-author-out value (from s6e8-diversity-beats-strength notebook):
  @boltuzamaki        45 arrays  value=4.2e-6/array  <- highest!
  @adarsh1077         22 arrays  value=2.6e-6/array
  @mohankrishnathalla  4 arrays  value=1.0e-6/array
  @najiama            14 arrays  value=0.7e-6/array
  @raykkretzschmar     5 arrays  value=0.4e-6/array
  @szymonkapiski      67 arrays  value=0.2e-6/array
  @beicicc            14 arrays  ~0 (but additive)
  @dariushafshar       7 arrays  ~0 (but include for completeness)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ================================================================================
# CONFIGURATION
# ================================================================================
OOF_DIR = Path("input/oof_libraries")

# Priority-ordered by leave-one-author-out value per array
DATASETS = [
    # (slug, description)
    ("boltuzamaki/s6e8-oof-prediction-library",
     "@boltuzamaki -- 45 arrays, highest value (4.2e-6/array)"),

    ("szymonkapiski/s6e8-oof-library-47-models",
     "@szymonkapiski -- 47 arrays (0.2e-6/array)"),

    ("adarsh1077/s6e8-adarsh-oof-library",
     "@adarsh1077 -- 22 models, 'diversity-beats-strength' author (2.6e-6/array)"),

    ("dariushafshar/s6e8-golem-oof-library",
     "@dariushafshar -- golem OOF library"),

    ("dariushafshar/s6e8-measured-findings-pack",
     "@dariushafshar -- measured findings pack (includes corrector arrays)"),

    ("najiama/predicting-smartphone-addiction-oof-submission-csv",
     "@najiama -- CSV blend pairs (0.7e-6/array)"),

    ("raykkretzschmar/s6e8-fm-lattice-blend-members",
     "@raykkretzschmar -- FM lattice blend members (0.4e-6/array)"),
]

# Additional beicicc and mohankrishnathalla datasets -- try common slugs
EXTRA_DATASETS = [
    # beicicc datasets (multiple, each ~2 arrays)
    ("beicicc/s6e8-cat-mlp-oof",         "@beicicc cat+mlp OOF"),
    ("beicicc/s6e8-lgb-dart-oof",        "@beicicc lgb-dart OOF"),
    ("beicicc/s6e8-xgb-oof",             "@beicicc xgb OOF"),
    ("beicicc/s6e8-extra-oof",           "@beicicc extra OOF"),
    # mohankrishnathalla
    ("mohankrishnathalla/s6e8-cat-oof",  "@mohankrishnathalla cat OOF"),
    ("mohankrishnathalla/s6e8-nn-oof",   "@mohankrishnathalla nn OOF"),
]


# Use sys.executable -m kaggle so the command always runs inside the current venv,
# regardless of whether 'kaggle' is on PATH or not.
KAGGLE_CMD = [sys.executable, "-m", "kaggle"]


def check_kaggle_cli():
    """Verify kaggle package is importable and credentials exist."""
    result = subprocess.run(KAGGLE_CMD + ["--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: kaggle package not found in this Python environment.")
        print(f"  Install with: {sys.executable} -m pip install kaggle")
        sys.exit(1)

    # Accept either the classic kaggle.json or the new OAuth access_token
    cred_json = Path.home() / ".kaggle" / "kaggle.json"
    cred_token = Path.home() / ".kaggle" / "access_token"
    env_token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")

    has_creds = cred_json.exists() or cred_token.exists() or bool(env_token)

    if not has_creds:
        print("\nERROR: Kaggle credentials not found. Choose one of these options:")
        print()
        print("  Option A (Recommended): OAuth login")
        print(f"    {sys.executable} -m kaggle auth login")
        print()
        print("  Option B: API token file")
        print("    1. Go to https://www.kaggle.com/settings -> API -> 'Create New Token'")
        print("    2. Save the downloaded kaggle.json to:")
        print(f"       {cred_json}")
        print()
        print("  Option C: Environment variable")
        print("    $env:KAGGLE_API_TOKEN = 'your_token_here'  # PowerShell")
        sys.exit(1)

    version_line = [l for l in result.stdout.splitlines() if "Kaggle" in l]
    print(f"[OK] {version_line[0] if version_line else 'kaggle CLI ready'}")
    if cred_json.exists():
        print(f"[OK] Credentials: {cred_json}")
    elif cred_token.exists():
        print(f"[OK] Credentials: {cred_token}")
    else:
        print("[OK] Credentials: environment variable")


def download_dataset(slug: str, description: str, force: bool = False):
    """Download and unzip a Kaggle dataset into OOF_DIR/<dataset-name>/."""
    dataset_name = slug.split("/")[-1]
    dest = OOF_DIR / dataset_name

    if dest.exists() and not force:
        npy = list(dest.rglob("*.npy"))
        parq = list(dest.rglob("*.parquet"))
        csv_oof = list(dest.rglob("*oof*.csv"))
        print(f"  [SKIP] {slug}")
        print(f"         Already exists: {len(npy)} .npy, {len(parq)} .parquet, {len(csv_oof)} OOF .csv")
        return True

    print(f"\n  [DL] Downloading {slug}")
    print(f"       {description}")

    dest.mkdir(parents=True, exist_ok=True)

    cmd = KAGGLE_CMD + [
        "datasets", "download",
        "--dataset", slug,
        "--path", str(dest),
        "--unzip",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "404" in stderr or "Not Found" in stderr or "doesn't exist" in stderr.lower():
            print(f"  [WARN] Not found (slug may be wrong, skipping): {slug}")
            shutil.rmtree(dest, ignore_errors=True)
            return False
        else:
            print(f"  [FAIL] Download failed: {stderr}")
            shutil.rmtree(dest, ignore_errors=True)
            return False

    npy = list(dest.rglob("*.npy"))
    parq = list(dest.rglob("*.parquet"))
    csv_oof = list(dest.rglob("*oof*.csv"))
    print(f"  [OK] {slug}: {len(npy)} .npy, {len(parq)} .parquet, {len(csv_oof)} OOF .csv")
    return True


def count_potential_pairs():
    """Count total OOF/test pairs available in the OOF_DIR."""
    if not OOF_DIR.exists():
        return 0

    pairs = set()

    for p in OOF_DIR.rglob("*.npy"):
        b = p.stem
        d = p.parent
        if b.startswith("oof_"):
            key = b[4:]
            if (d / f"test_{key}.npy").exists():
                pairs.add(f"npy:{d}/{key}")
        elif b.endswith("_oof"):
            key = b[:-4]
            if (d / f"{key}_test.npy").exists():
                pairs.add(f"npy:{d}/{key}")

    for p in OOF_DIR.rglob("*oof*.parquet"):
        d = p.parent
        b = p.name
        test_b = b.replace("oof", "test")
        if (d / test_b).exists() and test_b != b:
            pairs.add(f"parquet:{d}/{b}")

    for p in OOF_DIR.rglob("*_blend_oof_predictions.csv"):
        k = p.name.split("_")[0]
        sub_p = p.parent / f"{k}_blend_submission.csv"
        if sub_p.exists():
            pairs.add(f"csv:{p.parent}/{k}")

    return len(pairs)


def main():
    print("=" * 65)
    print("  S6E8 OOF Library Downloader")
    print("=" * 65)
    print(f"\n  Target directory: {OOF_DIR.resolve()}")
    print()

    check_kaggle_cli()
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'--'*32}")
    print("  Downloading priority datasets...")
    print(f"{'--'*32}")

    success, failed = 0, []
    for slug, desc in DATASETS:
        ok = download_dataset(slug, desc)
        if ok:
            success += 1
        else:
            failed.append(slug)

    print(f"\n{'--'*32}")
    print("  Trying extra datasets (may not all exist)...")
    print(f"{'--'*32}")

    for slug, desc in EXTRA_DATASETS:
        ok = download_dataset(slug, desc)
        if ok:
            success += 1

    n_pairs = count_potential_pairs()
    print(f"\n{'='*65}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*65}")
    print(f"  Datasets successfully downloaded/found: {success}")
    if failed:
        print(f"  Failed downloads: {len(failed)}")
        for s in failed:
            print(f"    - {s}")
    print(f"\n  Estimated OOF/test pairs available: ~{n_pairs}")
    print(f"  (Actual count after loading depends on shape validation)")
    print()
    print("  Next step: python main.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
