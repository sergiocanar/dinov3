import os
from os.path import join as path_join

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

from utils import load_json


# ----------------------------
# RLE -> RGBA (instance)
# ----------------------------
def rle_to_rgba_numpy(rle: dict) -> np.ndarray:
    """
    Convert COCO RLE to RGBA numpy (H,W,4)
    - RGB: white
    - Alpha: mask (0 bg, 255 fg)
    """
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    alpha = (m * 255).astype(np.uint8)

    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 1] = 255
    rgba[:, :, 2] = 255
    rgba[:, :, 3] = alpha
    return rgba


def write_rgba_png(out_path: str, rgba: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, rgba)


# ----------------------------
# Merge (union) RGBA alpha per category per image
# ----------------------------
def union_alpha(existing_rgba: np.ndarray, new_rgba: np.ndarray) -> np.ndarray:
    """
    Union operation for category masks: alpha = max(alpha1, alpha2)
    RGB stays white.
    """
    if existing_rgba is None:
        return new_rgba

    # Safety: shapes must match
    if existing_rgba.shape != new_rgba.shape:
        raise ValueError(f"Shape mismatch: {existing_rgba.shape} vs {new_rgba.shape}")

    out = existing_rgba.copy()
    out[:, :, 3] = np.maximum(existing_rgba[:, :, 3], new_rgba[:, :, 3])
    return out


# ----------------------------
# CSV creation helpers
# ----------------------------
def extract_cvs_labels_from_image_info(img_info: dict) -> tuple[int, int, int]:
    """
    Expected outputs:
      c1,c2,c3 in {0,1}
    """
    # Try common keys
    ds = img_info["ds"]
    c1, c2, c3 = ds
    return int(c1), int(c2), int(c3)
        

def write_csv(csv_path: str, rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image_path,masks_dir,c1,c2,c3\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


# ----------------------------
# Main conversion
# ----------------------------
def rle2masks_and_csv(
    json_info: dict,
    images_root: str,
    out_instances_root: str,
    out_categories_root: str,
    csv_out_path: str,
) -> None:
    """
    Writes:
      - instance masks:  out_instances_root/<stem>/<catid>_<annid>.png
      - category masks:  out_categories_root/<stem>/<catid>.png   (UNION of instances)
      - csv: image_path,masks_dir,y1,y2,y3
    """
    os.makedirs(out_instances_root, exist_ok=True)
    os.makedirs(out_categories_root, exist_ok=True)
    
    # Pre-index images by id for speed
    img_by_id = {im["id"]: im for im in json_info["images"]}

    # Accumulator: (img_id, cat_id) -> rgba array
    merged = {}

    # 1) Write instance masks and build merged per (img_id, cat_id)
    with tqdm(total=len(json_info["annotations"]), desc="RLE -> instance + merge", unit="annot") as pbar:
        for annot in json_info["annotations"]:
            #Extract info information
            ann_id = annot["id"]
            img_id = annot["image_id"]
            cat_id = annot["category_id"]
            rle = annot["segmentation"]

            #Extract image info
            img_info = img_by_id[img_id]
            file_name = img_info["file_name"]
            stem = file_name.split(".")[0]
            rgba = rle_to_rgba_numpy(rle)

            # instance path
            inst_dir = path_join(out_instances_root, stem)
            inst_name = f"{cat_id}_{ann_id}.png"
            inst_path = path_join(inst_dir, inst_name)
            write_rgba_png(inst_path, rgba)

            # merge per category
            key = (img_id, cat_id)
            if key not in merged:
                merged[key] = rgba
            else:
                merged[key] = union_alpha(merged[key], rgba)

            pbar.update(1)

    # 2) Write merged category masks
    with tqdm(total=len(merged), desc="Writing merged category masks", unit="catmask") as pbar:
        for (img_id, cat_id), rgba in merged.items():
            img_info = img_by_id[img_id]
            stem = img_info["file_name"].split(".")[0]
            cat_dir = path_join(out_categories_root, stem)
            cat_path = path_join(cat_dir, f"{cat_id}.png")
            write_rgba_png(cat_path, rgba)
            pbar.update(1)

    # 3) Create CSV rows
    rows = []
    with tqdm(total=len(json_info["images"]), desc="Building CSV", unit="img") as pbar:
        for img_id, img_info in img_by_id.items():
            file_name = img_info["file_name"]
            stem = os.path.splitext(os.path.basename(file_name))[0]

            image_path = path_join(images_root, file_name)
            masks_dir = path_join(out_categories_root, stem)

            # Extract 3 CVS labels (you may need to adapt this function!)
            y1, y2, y3 = extract_cvs_labels_from_image_info(img_info)

            rows.append([image_path, masks_dir, str(y1), str(y2), str(y3)])
            pbar.update(1)

    write_csv(csv_out_path, rows)
    print(f"[OK] Wrote CSV: {csv_out_path}")


if __name__ == "__main__":
    #Paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, "data")
    endo2023_dir = path_join(data_dir, "Endoscapes2023")
    annots_dir = path_join(endo2023_dir, "annotations")
    seg50_dir = path_join(annots_dir, "Seg50")

    all_coco_path = path_join(seg50_dir, "train_annotation_coco.json")
    all_info = load_json(json_path=all_coco_path)

    images_root = path_join(endo2023_dir, "frames")  

    out_instances = path_join(seg50_dir, "binary_masks_instances")
    out_categories = path_join(seg50_dir, "binary_masks_categories")
    csv_out = path_join(seg50_dir, "cvs_labels.csv")

    rle2masks_and_csv(
        json_info=all_info,
        images_root=images_root,
        out_instances_root=out_instances,
        out_categories_root=out_categories,
        csv_out_path=csv_out,
    )
