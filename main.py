import os
from os.path import join as path_join


import wandb
import argparse
import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments, set_seed

from torchmetrics.classification import MultilabelAveragePrecision
from cvs_datasets.CVS_Dataset import CVSData, move_last_n_videos
from CVS_models.SurgicalDINO_CVS import SurgicalDINOForCVS
from utils import cvs_data_collator, save_json, cvs_dino_collator  


def cvs_data_collator(batch, feature_extractor):
    images = [img for img, label, video_name, frame_id, metadata in batch]
    labels = torch.stack([label for img, label, video_name, frame_id, metadata in batch], dim=0)   # [B, 3]

    enc = feature_extractor(images, return_tensors="pt")
    pixel_values = enc["pixel_values"]  # [B, 3, 224, 224]

    # Return only what the model forward expects:
    return {"pixel_values": pixel_values, "labels": labels}


@torch.no_grad()
def make_predictions_json(trainer, dataset, out_path, split_name="test", threshold=0.5):
    """
    Saves per-sample predictions:
      video_name, frame_id, logits, probs, pred (0/1), label (0/1)
    """
    pred = trainer.predict(test_dataset=dataset)
    logits = pred.predictions
    labels = pred.label_ids

    probs = 1 / (1 + np.exp(-logits[0]))  # sigmoid
    preds = (probs >= threshold).astype(int)

    records = []
    for i in range(len(dataset)):
        # IMPORTANT: This assumes your dataset returns:
        # (img, label, video_name, frame_id, meta)
        img, label_t, video_name, frame_id, meta = dataset[i]
        records.append({
            "split": split_name,
            "video_name": video_name,
            "frame_id": int(frame_id),
            "probs": probs[i].tolist(),
            "pred": preds[i].tolist(),
            "label": labels[i].astype(int).tolist(),
        })

    save_json(records, out_path)
    print(f"Saved json to: {out_path}")


# -------------------------
# Metrics
# -------------------------
def multilabel_map_torch(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Simple mAP wrapper.
    If you already have mAP_metric elsewhere, replace this.
    probs: (N, C) float in [0,1]
    labels: (N, C) int in {0,1}
    """
    # If you have torchmetrics available:
    try:
        metric = MultilabelAveragePrecision(num_labels=probs.shape[1], average="macro")
        return float(metric(probs.cpu(), labels.cpu()).item())
    except Exception:
        # Fallback: return NaN if torchmetrics isn't installed
        return float("nan")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.sigmoid(torch.tensor(logits))
    labels_t = torch.tensor(labels).int()
    mAP = multilabel_map_torch(probs, labels_t)
    return {"mAP": mAP}


# -------------------------
# Arg parser
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("DINOv3 CVS training (HF Trainer)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    p.add_argument("--run_name", type=str, default="run1")
    p.add_argument("--output_dir", type=str, default="outputs/dino_cvs")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--train-val", action="store_true", help="Train on train+val (no val during training).")

    # Model knobs
    p.add_argument("--backbone_size", type=str, default="large", choices=["small","base","large","giant"])
    p.add_argument("--use_layers", type=str, default="4", choices=["1","4"])
    p.add_argument("--head_type", type=str, default="mlp", choices=["mlp","transformer"])
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=14)

    # Transformer head knobs (only used if head_type=transformer)
    p.add_argument("--tf_depth", type=int, default=2)
    p.add_argument("--tf_heads", type=int, default=8)
    p.add_argument("--tf_dropout", type=float, default=0.1)
    p.add_argument("--tf_mlp_ratio", type=int, default=4)

    # Predictions output
    p.add_argument("--pred_threshold", type=float, default=0.5)
    return p


# -------------------------
# Main training flow
# -------------------------
def main(parser: argparse.ArgumentParser, main_dir: str, data_dir: str, weights_dir: str):
    
    #Parse arguments
    args = parser.parse_args()
    
    #Set seed
    set_seed(args.seed)

    if args.wandb:
        wandb.init(project="DINO CVS")
        # unique run name + dirs
        # run_name = args.run_name or f"vit_lr{lr:.0e}"
        run_name = wandb.run.name 
        safe_run_name = run_name.replace("/", "_")
        safe_run_name = wandb.run.name
        lr = float(wandb.config.lr)
        head_type = str(wandb.config.head_type)

    else:
        os.environ["WANDB_DISABLED"] = "true"
        safe_run_name = args.run_name
        lr = args.lr
        head_type = args.head_type

    output_dir = path_join(main_dir, "outputs", "DINO_CVS", safe_run_name)
    logging_dir = path_join(output_dir, "logs")
    results_dir = path_join(output_dir, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    train_dir = path_join(data_dir, "train")
    train_frames_dir = path_join(train_dir, "frames")
    train_labels_dir = path_join(train_dir, "labels")
    
    test_dir = path_join(data_dir, "test")
    test_frames_dir = path_join(test_dir, "frames")
    test_labels_dir = path_join(test_dir, "labels")
    
    val_dir = path_join(data_dir, "val")
    val_frames_dir = path_join(val_dir, "frames")
    val_labels_dir = path_join(val_dir, "labels")

    #Create val set if not exists
    if not os.path.exists(val_dir):
        move_last_n_videos(train_frames_dir, train_labels_dir, data_dir, n=200)
    else:
        pass
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # If your dataset returns PIL images:
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),   # or transforms.Resize(256) + CenterCrop(224)
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    train_dataset = CVSData(
        frames_path=train_frames_dir,
        labels_path=train_labels_dir,
        transform=dino_transform
    )
    
    val_dataset = CVSData(
        frames_path=val_frames_dir,
        labels_path=val_labels_dir,
        transform=dino_transform
    )
    
    test_dataset = CVSData(
        frames_path=test_frames_dir,
        labels_path=test_labels_dir,
        transform=dino_transform
    )

    if args.train_val:
        # merge train+val for final training
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = None

    # ---- Model ----
    model = SurgicalDINOForCVS(
        backbone_size=args.backbone_size,
        r=args.lora_r,
        lora_layer=None,                # or pass a list of block indices
        num_labels=3,
        use_layers=args.use_layers,
        head_type=head_type,
        tf_depth=args.tf_depth,
        tf_heads=args.tf_heads,
        tf_mlp_ratio=args.tf_mlp_ratio,
        tf_dropout=args.tf_dropout,
        img_size=args.img_size,
        patch_size=args.patch_size,
        weights_dir=weights_dir
    )

    # ---- TrainingArguments ----
    do_eval = (val_dataset is not None) and (not args.eval_only)
    eval_strategy = "epoch" if do_eval else "no"
    out_dir = os.path.join(output_dir, safe_run_name, "preds")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        run_name=safe_run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=do_eval,
        save_total_limit=1,
        metric_for_best_model="mAP",
        greater_is_better=True,
        report_to=["wandb"] if args.wandb else [],
        remove_unused_columns=False,  # IMPORTANT: we use custom keys in the collator
        dataloader_num_workers=args.num_workers,
        fp16=torch.cuda.is_available(),  # optional; if bf16 supported you can switch
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if not args.eval_only else None,
        eval_dataset=val_dataset if do_eval else None,
        data_collator=cvs_dino_collator,
        compute_metrics=compute_metrics if do_eval else None,
    )

    # ---- Train / Eval ----
    if not args.eval_only:
        trainer.train()

    # ---- Save prediction JSONs ----
    # (You can do for train/val/test if you want.)
    os.makedirs(out_dir, exist_ok=True)
    make_predictions_json(
        trainer=trainer,
        dataset=test_dataset,
        out_path=os.path.join(out_dir, "test_predictions.json"),
        split_name="test",
        threshold=args.pred_threshold,
    )

    if val_dataset is not None:
        make_predictions_json(
            trainer=trainer,
            dataset=val_dataset,
            out_path=os.path.join(out_dir, "val_predictions.json"),
            split_name="val",
            threshold=args.pred_threshold,
        )


if __name__ == "__main__":

    #Build ArgumentParser
    parser = build_parser()    
    
    #Paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, "data")
    sages_2024_dir = path_join(data_dir, "SAGES_2024")
    
    main(parser=parser,
         main_dir=this_dir,
         data_dir=sages_2024_dir,
         weights_dir="/home/scanar/endovis/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
