"""Read LeRobot v3.0 dataset metadata and parquet rows without importing lerobot.

lerobot's own dataset class refuses to open a local copy unless every video is
present and otherwise falls back to the Hub; reading the files directly avoids
that and keeps Kaggle images light.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetMeta:
    root: Path
    info: dict
    episodes: pd.DataFrame  # one row per episode, sorted by episode_index
    cams: list[str]  # e.g. ["observation.images.top", "observation.images.right"]
    tasks: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path) -> "DatasetMeta":
        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        files = sorted(glob.glob(str(root / "meta/episodes/chunk-*/*.parquet")))
        eps = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)
        cams = [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")]
        tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
        return cls(root, info, eps, cams, tasks)

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    def __len__(self) -> int:
        return len(self.episodes)

    def data_path(self, ep: pd.Series) -> Path:
        return self.root / self.info["data_path"].format(chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"]))

    def video_path(self, ep: pd.Series, cam: str) -> Path:
        return self.root / self.info["video_path"].format(
            video_key=cam, chunk_index=int(ep[f"videos/{cam}/chunk_index"]), file_index=int(ep[f"videos/{cam}/file_index"])
        )

    def video_offset(self, ep: pd.Series, cam: str) -> float:
        """Episodes are concatenated in one mp4; this is where this episode starts in the file's clock."""
        return float(ep[f"videos/{cam}/from_timestamp"])

    def task_string(self, task_index: int) -> str:
        return str(self.tasks.index[task_index]) if len(self.tasks) else ""

    def rows(self, ep: pd.Series) -> dict[str, np.ndarray | int]:
        """Timestamps, state, action for one episode as numpy arrays."""
        df = pd.read_parquet(self.data_path(ep), columns=["episode_index", "timestamp", "observation.state", "action", "task_index"])
        df = df[df["episode_index"] == int(ep["episode_index"])].reset_index(drop=True)
        return {
            "timestamp": df["timestamp"].to_numpy(np.float32),
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "action": np.stack(df["action"].to_numpy()).astype(np.float32),
            "task_index": int(df["task_index"].iloc[0]),
        }
