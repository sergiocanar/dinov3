import os
import argparse
from os.path import join as path_join

import csv
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelAveragePrecision

from tqdm import tqdm
import torchvision.transforms.functional as TF


def DinoCVS_parser():
    parser = argparse.ArgumentParser(description="DINOv3 CVS Baseline")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    
    return parser


# ----------------------------
# Config
# ----------------------------
PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

MODEL_NAME = "dinov3_vitb16"
DINOV3_LOCATION = "/home/scanar/endovis/models/dinov3"
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 12
NUM_EPOCHS = 10
LR_BACKBONE = 5e-6
LR_HEAD = 1e-3
WD_BACKBONE = 0.05
WD_HEAD = 1e-4
NUM_WORKERS = 8


# If you know exact blocks, set it; otherwise we try to infer.
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vitsp16": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vithp16": 32,
    "dinov3_vit7b16": 40,
}


# ----------------------------
# Resize transform (from your notebook)
# ----------------------------
def resize_transform(
    pil_img: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = pil_img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    return TF.to_tensor(TF.resize(pil_img, (new_h, new_w)))


# ----------------------------
# Dataset (CSV)
# ----------------------------
class CVSBaselineDataset(Dataset):
    """
    CSV rows:
      image_path,y1,y2,y3
    """
    def __init__(self, csv_path: str):
        self.items: List[Tuple[str, torch.Tensor]] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # allow header or not
            for row in reader if header and "image_path" in header[0] else [header] + list(reader):
                if row is None:
                    continue
                if len(row) < 4:
                    continue
                img_path = row[0].strip()
                y = torch.tensor(list(map(int, row[1:4])), dtype=torch.float32)
                self.items.append((img_path, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, y = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return img, y


def collate_fn(batch):
    imgs, ys = zip(*batch)
    return list(imgs), torch.stack(ys, dim=0)


# ----------------------------
# Head
# ----------------------------
class CVSHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)  # logits (B,3)


# ----------------------------
# Load DINOv3
# ----------------------------
def load_dinov3(train_encoder: bool = True):
    source = "local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
    repo_or_dir = DINOV3_LOCATION if source == "local" else DINOV3_GITHUB_LOCATION

    model = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=MODEL_NAME,
        source=source,
    ).to(DEVICE)

    if train_encoder:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model


# ----------------------------
# Feature extraction (no masks): global average pool over patch features
# ----------------------------
def batch_to_embeddings(dinov3_model, images_pil: List[Image.Image], n_layers: int) -> torch.Tensor:
    """
    Returns (B, D) embeddings using last-layer patch feature map average pooling.
    """
    batch_t = []
    for img in images_pil:
        t = resize_transform(img)  # (3,H,W) in [0,1]
        t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        batch_t.append(t)
    x = torch.stack(batch_t, dim=0).to(DEVICE)  # (B,3,H,W)

    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=(DEVICE == "cuda")):
        feats = dinov3_model.get_intermediate_layers(x, n=range(n_layers), reshape=True, norm=True)
        # feats[-1]: (B, D, H', W')
        fmap = feats[-1]  # keep on GPU for speed
        emb = fmap.mean(dim=(2, 3))  # (B, D)

    return emb


# ----------------------------
# Metrics
# ----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    f1s = []
    for k in range(y_true.shape[1]):
        tp = np.sum((y_true[:, k] == 1) & (y_pred[:, k] == 1))
        fp = np.sum((y_true[:, k] == 0) & (y_pred[:, k] == 1))
        fn = np.sum((y_true[:, k] == 1) & (y_pred[:, k] == 0))
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        f1s.append(f1)
    return float(np.mean(f1s))



@torch.no_grad()
def evaluate(
    dinov3_model,
    head,
    loader,
    criterion,
    n_layers: int,
    num_labels: int = 3,
):
    head.eval()
    total_loss = 0.0

    # Torchmetrics object (on device)
    map_metric = MultilabelAveragePrecision(
        num_labels=num_labels,
        average="macro",        # mean over labels
    ).to(DEVICE)

    # (Optional) per-class AP
    map_per_class = MultilabelAveragePrecision(
        num_labels=num_labels,
        average=None,
    ).to(DEVICE)

    for imgs, y in tqdm(loader, desc="Eval", leave=False):
        y = y.to(DEVICE)                    # (B,3) in {0,1}
        emb = batch_to_embeddings(dinov3_model, imgs, n_layers=n_layers)
        logits = head(emb)

        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)

        probs = torch.sigmoid(logits)       # (B,3) in [0,1]

        # Update metrics
        map_metric.update(probs, y.int())
        map_per_class.update(probs, y.int())

    mean_ap = map_metric.compute().item()               # scalar
    ap_per_class = map_per_class.compute().cpu().numpy()  # (3,)

    return (
        total_loss / len(loader.dataset),
        mean_ap,
        ap_per_class,
    )


def train_one_epoch(dinov3_model, head, loader, optimizer, criterion, n_layers: int):
    dinov3_model.train()
    head.train()
    total_loss = 0.0

    for imgs, y in tqdm(loader, desc="Train", leave=False):
        y = y.to(DEVICE)

        emb = batch_to_embeddings(dinov3_model, imgs, n_layers=n_layers)
        logits = head(emb)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(dinov3_model.parameters()) + list(head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)

    return total_loss / len(loader.dataset)


# ----------------------------
# Main
# ----------------------------
def main(train_csv: str, val_csv: str, train_encoder: bool,train_or_test: str = "train"):
    dinov3_model = load_dinov3(train_encoder=train_encoder)
    n_layers = MODEL_TO_NUM_LAYERS.get(MODEL_NAME, 12)

    # Infer D
    dummy = Image.new("RGB", (640, 480))
    emb = batch_to_embeddings(dinov3_model, [dummy], n_layers=n_layers)
    d_in = int(emb.shape[1])
    print(f"[INFO] Using {MODEL_NAME} | n_layers={n_layers} | d_in={d_in}")
    
    head = CVSHead(d_in=d_in, hidden=512).to(DEVICE)
    if train_or_test == "test":
        ckpt = torch.load("dinov3_cvs_finetuned.pth", map_location=DEVICE)
        dinov3_model.load_state_dict(ckpt["dinov3"], strict=True)
        head.load_state_dict(ckpt["head"], strict=True)
        dinov3_model.eval()
        head.eval()

        val_ds = CVSBaselineDataset(val_csv)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn)

        criterion = nn.BCEWithLogitsLoss()
        va_loss, ap, ap_per_class = evaluate(dinov3_model, head, val_loader, criterion, n_layers=n_layers)
        print(f"Test Loss={va_loss:.4f} | mAP={ap} | AP per class={ap_per_class}")
        
    else:
        head = CVSHead(d_in=d_in, hidden=512).to(DEVICE)

        train_ds = CVSBaselineDataset(train_csv)
        val_ds = CVSBaselineDataset(val_csv)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            [
                {"params": dinov3_model.parameters(), "lr": LR_BACKBONE, "weight_decay": WD_BACKBONE},
                {"params": head.parameters(), "lr": LR_HEAD, "weight_decay": WD_HEAD},
            ]
        )

        best_ap = -1.0
        for epoch in range(NUM_EPOCHS):
            tr_loss = train_one_epoch(dinov3_model, head, train_loader, optimizer, criterion, n_layers=n_layers)
            va_loss, ap, ap_per_class = evaluate(dinov3_model, head, val_loader, criterion, n_layers=n_layers)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} " f"| mAP={ap} | AP per class={ap_per_class}")

            if ap > best_ap:
                best_ap = ap
                torch.save(
                    {
                        "head": head.state_dict(),
                        "dinov3": dinov3_model.state_dict(),
                        "model_name": MODEL_NAME,
                        "n_layers": n_layers,
                    },
                    "dinov3_cvs_finetuned.pth"
                )


if __name__ == "__main__":
    
    parser = DinoCVS_parser()
    args = parser.parse_args()    
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, "data")
    endoscapes_dir = path_join(data_dir, "Endoscapes2023")
    annots_dir = path_join(endoscapes_dir, "annotations")
    cvs_dir = path_join(annots_dir, "CVS201")
    
    TRAIN_CSV = path_join(cvs_dir, "train.csv")
    VAL_CSV = path_join(cvs_dir, "test.csv")
    main(TRAIN_CSV, VAL_CSV, args.mode)
