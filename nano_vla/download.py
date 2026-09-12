"""Robust per-file download of a HF dataset repo (retries each file until present)."""
import argparse, os, time
from huggingface_hub import HfApi, hf_hub_download

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ap = argparse.ArgumentParser()
ap.add_argument("repo_id"); ap.add_argument("local_dir")
args = ap.parse_args()
files = [s.rfilename for s in HfApi().dataset_info(args.repo_id).siblings]
print(len(files), "files")
missing = [f for f in files if not os.path.exists(os.path.join(args.local_dir, f))]
while missing:
    still = []
    for f in missing:
        for attempt in range(6):
            try:
                hf_hub_download(args.repo_id, f, repo_type="dataset", local_dir=args.local_dir); break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        else:
            still.append(f)
    print(f"{len(files) - len(still)}/{len(files)} present", flush=True)
    missing = still
print("DONE")
