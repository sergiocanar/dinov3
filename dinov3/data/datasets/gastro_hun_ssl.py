import logging
import os
from typing import Any

import pandas as pd
import torch

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


def _seq_dir_from_mp4_filename(filename: str) -> str:
    """
    Convert all_vid_sequences.csv filename to sequence frame directory.

    e.g. "video_080/video_080_0000_L1.mp4"
              -> relative path "video_080/0000_L1"
    """
    vid_id, mp4_name = filename.split("/", 1)
    # strip "<vid_id>_" prefix and ".mp4" suffix
    seq_name = mp4_name.removeprefix(f"{vid_id}_").removesuffix(".mp4")
    return f"{vid_id}/{seq_name}"


class GastroHUNSSL(ExtendedVisionDataset):
    """
    Flat frame-level SSL dataset for GastroHUN H. Pylori sequences.

    Collects all frames within ±(window_secs/2 * fps) of the keyframe for
    every sequence in the chosen CSV split.  Produces a flat list of image
    paths — no labels — suitable for DINOv3 SSL training.

    Two CSV formats are handled automatically:

    1. all_vid_sequences.csv  (split="all", default)
       columns: num_patient, filename, central_frame_idx_in_seq, …
       filename format: "video_080/video_080_0000_L1.mp4"
       Keyframe index read directly from ``central_frame_idx_in_seq``.

    2. fold CSVs  (split="fold1_train", "fold2_val", etc.)
       columns: num_patient, filename, HP, OLGA, …
       filename format: "video_080/0000_L1.pth"
       Keyframe index loaded from the corresponding .pth in seq_features_dir.

    Expected directory layout (relative to root):
        sequences/labels/<split>.csv
        seq_features/<video>/<seq>.pth   (only for fold-CSV mode)
        sequences/frames/<video>/<seq>/  JPEG frames

    Args:
        root         : GastroHUN dataset root
        split        : "all" → sequences/labels/all_vid_sequences.csv
                       anything else → sequences/labels/<split>.csv  (fold mode)
        window_secs  : seconds of context around the keyframe (default 10 s)
        fps          : effective fps of saved frames (default 15.0)
    """

    def __init__(
        self,
        root: str,
        split: str = "all",
        window_secs: float = 10.0,
        fps: float = 15.0,
        **kwargs,
    ) -> None:
        super().__init__(root=root, **kwargs)

        frames_root = os.path.join(root, "sequences", "frames")
        seq_features_dir = os.path.join(root, "seq_features")

        if split == "all":
            csv_path = os.path.join(root, "sequences", "labels", "all_vid_sequences.csv")
        else:
            csv_path = os.path.join(root, "sequences", "labels", f"{split}.csv")

        half_w = int(window_secs * fps / 2)  # e.g. 10s * 15fps / 2 = 75 frames

        df = pd.read_csv(csv_path)
        all_csv_mode = "central_frame_idx_in_seq" in df.columns

        self._frame_paths: list[str] = []

        for _, row in df.iterrows():
            fname = row["filename"]

            if all_csv_mode:
                kf_idx = int(row["central_frame_idx_in_seq"])
                rel_seq_dir = _seq_dir_from_mp4_filename(fname)
            else:
                # fold-CSV mode: load .pth to get keyframe_idx
                pth_path = os.path.join(seq_features_dir, fname)
                try:
                    data = torch.load(pth_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    logger.warning(f"Could not load {pth_path}: {e} — skipping")
                    continue
                kf_idx = int(data["keyframe_idx"])
                rel_seq_dir = fname.replace(".pth", "")

            seq_dir = os.path.join(frames_root, rel_seq_dir)
            if not os.path.isdir(seq_dir):
                logger.warning(f"Frame directory not found: {seq_dir} — skipping")
                continue

            all_frames = sorted(f for f in os.listdir(seq_dir) if not f.startswith("."))
            n_frames = len(all_frames)

            lo = max(0, kf_idx - half_w)
            hi = min(n_frames - 1, kf_idx + half_w)

            for i in range(lo, hi + 1):
                self._frame_paths.append(os.path.join(seq_dir, all_frames[i]))

        logger.info(
            f"GastroHUNSSL [{split}]: {len(df)} sequences → "
            f"{len(self._frame_paths):,} frames  (window ±{half_w} frames @ {fps} fps)"
        )

    def get_image_data(self, index: int) -> bytes:
        with open(self._frame_paths[index], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return ()

    def __len__(self) -> int:
        return len(self._frame_paths)
