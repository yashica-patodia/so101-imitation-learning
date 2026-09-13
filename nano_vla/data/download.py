"""Robust per-file download of a Hugging Face dataset repo.

The Hub drops connections often enough that snapshot_download gives up; this
retries each file until every one listed in the repo is present locally.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def download(repo_id: str, local_dir: str | Path, rounds: int = 20, per_file_tries: int = 6) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    local_dir = Path(local_dir)
    files = [s.rfilename for s in HfApi().dataset_info(repo_id).siblings]
    missing = [f for f in files if not (local_dir / f).exists()]
    print(f"{repo_id}: {len(files)} files, {len(missing)} to fetch", flush=True)
    for r in range(rounds):
        if not missing:
            break
        still = []
        for f in missing:
            for attempt in range(per_file_tries):
                try:
                    hf_hub_download(repo_id, f, repo_type="dataset", local_dir=str(local_dir))
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(2 * (attempt + 1))
            else:
                still.append(f)
        print(f"  round {r}: {len(files) - len(still)}/{len(files)} present", flush=True)
        missing = still
    if missing:
        raise RuntimeError(f"{len(missing)} files still missing after {rounds} rounds")
    return local_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("local_dir")
    a = ap.parse_args()
    download(a.repo_id, a.local_dir)
    print("DONE")
