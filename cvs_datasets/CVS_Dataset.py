import os
from os.path import join as path_join

import glob
import shutil
from statistics import mode

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def move_last_n_videos(train_frames_dir: str, train_labels_dir: str, data_dir: str, n: int=200):
    val_frames_dir = path_join(data_dir, "val", "frames")
    val_labels_dir = path_join(data_dir, "val", "labels")
    os.makedirs(val_frames_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    video_names = sorted([
        d for d in os.listdir(train_frames_dir)
        if os.path.isdir(path_join(train_frames_dir, d))
    ])[-n:]

    for v in video_names:
        shutil.move(path_join(train_frames_dir, v), path_join(val_frames_dir, v))
        # labels might not exist for every video
        src_lab = path_join(train_labels_dir, v)
        if os.path.exists(src_lab):
            shutil.move(src_lab, path_join(val_labels_dir, v))

    print(f"[OK] moved {len(video_names)} videos train -> val")

class CVSData(Dataset):
    """Dataloader that returns annotated frames and their
    corresponding labels (majority), video names, and frame ids.
      Args:
        frames_path (str):   Path to folder containing the extracted frames
        labels_path (str):   Path to folder containing a label csv file for each video

      Returns:
        img, label, video_name, frame_id, metadata
    """

    def __init__(self, frames_path, labels_path, transform=None):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.transform = transform
        search_pattern = os.path.join(frames_path, "**", "*.jpg")
        frames = sorted(glob.glob(search_pattern, recursive=True))
        self.image_paths = [file for file in frames if self.is_annotated(file)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        video_name = img_path.split("/")[-2]
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        frame_name = frame_name.split('_')[-1]
        frame_id = int(frame_name)
        # frame_id = int(img_path.split("/")[-1].replace(".jpg", "")) - 1
        label_path = os.path.join(self.labels_path, f"{video_name}/frame.csv")
        video_label_path = os.path.join(self.labels_path, f"{video_name}/video.csv")

        # Load image
        img = Image.open(img_path)

        # Load c1,c2,c3 label
        label_df = pd.read_csv(label_path)
        video_label_df = pd.read_csv(video_label_path)
        confidences = [float(video_label_df[f'confidence_rater{i+1}'].iloc[0]) for i in range(3)]
        label_df = label_df[label_df["frame_id"] == frame_id]
        c1, c2, c3 = (
            self.majority_vote(label_df, "c1"),
            self.majority_vote(label_df, "c2"),
            self.majority_vote(label_df, "c3"),
        )
        ca_c1 = self.confidence_multiplexed_majority_vote(label_df, "c1",confidences)
        ca_c2 = self.confidence_multiplexed_majority_vote(label_df, "c2",confidences)
        ca_c3 = self.confidence_multiplexed_majority_vote(label_df, "c3",confidences)
        label = torch.as_tensor([ca_c1, ca_c2, ca_c3], dtype=torch.float32)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        metadata = {}
        metadata["raw_labels"] = {}
        for key in ["c1", "c2", "c3"]:
            metadata["raw_labels"][key] = self.raw_labels(label_df, key)
        metadata["confidence_aware_labels"] = {}
        ca_labels = [ca_c1,ca_c2,ca_c3]
        for i,key in enumerate(["c1", "c2", "c3"]):
            metadata["confidence_aware_labels"][key] = ca_labels[i]
        
        return img, label, video_name, frame_id, metadata

    def is_annotated(self, fname):
        frame_name = os.path.splitext(os.path.basename(fname))[0]
        frame_name = frame_name.split('_')[-1]
        frame_id = int(frame_name)
        # frame_id = int(os.path.splitext(os.path.basename(fname))[0]) - 1
        return frame_id % 150 == 0

    def raw_labels(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]

        return labels

    def majority_vote(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]
        return mode(labels)

    def confidence_multiplexed_majority_vote(self, row, category,confidences):
        labels = np.array([
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ])
        labeler_confidences = np.array(confidences)
        confidence_aware_label = 0.5+1/3.0*np.dot(labels-0.5,labeler_confidences)
        return confidence_aware_label


class HFViTDataset(Dataset):
    """
    Wrap CVSData output:
      (PIL, label_tensor[3], video_name, frame_id, metadata)
    into a dict that Trainer/data_collator can handle.
    """
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label, video_name, frame_id, meta = self.base[idx]
        return {
            "image": img,                 # PIL
            "labels": label.float(),      # tensor [3]
            "video_name": video_name,
            "frame_id": frame_id,
            "metadata": meta
        }
