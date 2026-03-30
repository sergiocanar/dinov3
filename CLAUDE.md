# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository adapts **DINOv3** (Meta's ViT-7B vision transformer) for **Critical View of Safety (CVS)** classification in surgical videos. CVS is a 3-class multilabel prediction task on frames from the SAGES 2024 cholecystectomy dataset.

The `dinov3/` directory contains the upstream DINOv3 library from Meta. The custom CVS adaptation lives in `CVS_models/`, `main.py`, `evaluate.py`, `inference.py`, and `utils.py`.

## Commands

**Training (standalone):**
```bash
python main.py --run_name my_run --epochs 20 --batch_size 8 --lr 2e-5 --weight_decay 5e-4
```

**Training with W&B:**
```bash
CUDA_VISIBLE_DEVICES=2 python main.py --wandb
```

**W&B hyperparameter sweep:**
```bash
wandb sweep DINO_CVS.yaml
wandb agent <sweep_id>
```

**Inference on test set (from HF checkpoint dir):**
```bash
python inference.py --checkpoint outputs/DINO_CVS/<run_name>/checkpoint-<step> \
  --weights_dir /home/scanar/endovis/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

**Evaluate predictions from JSON:**
```bash
python evaluate.py --pred_path outputs/DINO_CVS/<run_name>/<run_name>/preds/test_predictions.json
```

**Environment setup:**
```bash
conda env create -f conda.yaml
conda activate dinov3
# Also install: transformers wandb safetensors torchmetrics
```

## Architecture

### Primary model: `CVS_models/DINOv3_CVS.py` — `DinoV3_LC_LastAttnFT_CVS`

Loaded via `torch.hub.load(..., source="local")` from the `dinov3/` package. Strategy:
- Backbone (ViT-7B) is **fully frozen** except the last `k` attention blocks + their norms (`last_k_blocks=4` by default).
- Pretrained ImageNet linear head (1000-class) is **frozen**.
- A trainable **CVS head** maps ImageNet logits → 3 CVS labels: `LayerNorm → Linear(1000→256) → GELU → Dropout → Linear(256→3)`.
- Loss: `BCEWithLogitsLoss` with optional per-class `pos_weight`.

### Alternative model: `CVS_models/SurgicalDINO_CVS.py` — `SurgicalDINOForCVS`

Uses LoRA adapters on the backbone with a configurable head (`mlp` or `transformer`). The transformer head uses cross-attention with learned query tokens over patch tokens. Currently commented out in `main.py` but used by `inference.py`.

### Training: `main.py`

Uses HuggingFace `Trainer` with a custom `cvs_dino_collator` (from `utils.py`). Checkpoints are saved in HF safetensors format. The validation set is carved from training videos via `move_last_n_videos`. Metrics: `mAP_macro`, `mAP_micro`, `mAP_weighted`, per-class AP. Best model selected by `mAP_macro`.

### Data

`CVSData` (from external `cvs_datasets` package) returns `(image_tensor, label_tensor, video_name, frame_id, meta)`. Labels support soft/confidence-aware float values — binarized at 0.5 for metric computation but used as-is for BCE loss.

### Evaluation: `evaluate.py`

Standalone evaluation from a predictions JSON. Computes exact match accuracy, macro F1, per-class mAP, and Brier score (supports confidence-aware labels). Expected JSON format: list of dicts with `video_name`, `frame_id`, `label` (int list), `probs` (float list), optionally `confidence_aware_label`.

## Key Paths

- **Weights:** `/home/scanar/endovis/models/dinov3/weights/`
  - `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` — backbone
  - `dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth` — ImageNet head
- **Dataset:** `data/SAGES_2024/{train,val,test}/{frames,labels}/`
- **Outputs:** `outputs/DINO_CVS/<run_name>/` — checkpoints, logs, prediction JSONs

## Hub Loading

The `dinov3/` package is used as a local `torch.hub` source. Hub entrypoints (e.g., `dinov3_vit7b16_lc`) are defined in `hubconf.py`. The `repo_dir` argument in `DinoV3_LC_LastAttnFT_CVS` must point to this repo root (defaults to `/home/scanar/endovis/models/dinov3/`). Do **not** register the full hub model wrapper as `self._hub_model` — only store `m.backbone` and `m.linear_head` to avoid duplicate tensor names in the state dict.

## W&B Sweep

`DINO_CVS.yaml` defines a Bayesian sweep over `lr` and `weight_decay`. When `--wandb` is passed, hyperparameters are read from `wandb.config` instead of CLI args. W&B project: `DINOv3-CVS-7B`. Target metric: `eval/mAP`.
