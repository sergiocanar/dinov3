import os
from os.path import join as path_join

import json
from collections import defaultdict

import numpy as np
from utils import load_json, save_json
from PIL import Image
from pycocotools import mask as mask_utils



def anns_to_instance_mask_u8(ann_list, height: int, width: int, *, overlap="last", sort_by_area=False):
    """
    Build a uint8 instance mask for one image:
      0 = background
      1..N = instances (contiguous)
    """
    if sort_by_area:
        # Useful for overlap handling; pick what you want
        ann_list = sorted(ann_list, key=lambda a: a.get("area", 0))

    inst = np.zeros((height, width), dtype=np.uint16)
    next_id = 1

    for ann in ann_list:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        # segmentation can be polygons (list) or RLE (dict)
        if isinstance(seg, list):
            # polygons -> merged RLE
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            rle = seg
        else:
            raise ValueError(f"Unsupported segmentation type: {type(seg)}")

        m = mask_utils.decode(rle)  # (H,W) or (H,W,N)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        m = m.astype(bool)

        if not np.any(m):
            continue

        if overlap == "last":
            inst[m] = next_id
        elif overlap == "first":
            inst[m & (inst == 0)] = next_id
        else:
            raise ValueError("overlap must be 'last' or 'first'")

        next_id += 1
        if next_id > 256:
            raise ValueError(
                "More than 255 instances in one image -> cannot store in uint8. "
                "Use uint16 output or filter instances."
            )

    return inst.astype(np.uint8)


def instance_mask_to_rgb(mask_u8: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simple RGB visualization for a uint8 instance mask."""
    if mask_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 mask.")
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[mask_u8]


def coco_json_to_png_masks(
    coco_json_path: str,
    out_dir: str,
    *,
    use_filename_stem: bool = True,
    viz_dir: str | None = None,
    overlap: str = "last",
    sort_by_area: bool = False,
    skip_missing_size: bool = False):
    
    """
    Convert a full COCO JSON to per-image PNG instance masks.

    Saves:
      - out_dir/<file_stem>.png   (or image_id.png if use_filename_stem=False)

    Args:
      coco_json_path: path to COCO annotations json.
      out_dir: directory to save grayscale instance masks.
      use_filename_stem: if True, uses images[i]["file_name"] (stem) for output naming.
                         otherwise uses image_id (e.g., 123.png).
      viz_dir: if provided, saves RGB visualizations there with suffix _viz.png
      overlap: "last" or "first" overlap rule
      sort_by_area: if True, sorts anns by 'area' before assigning IDs
      skip_missing_size: if True, skips images missing height/width; else raises.
    """
    os.makedirs(out_dir, exist_ok=True)
    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    coco = load_json(coco_json_path)

    images = coco["images"]
    anns = coco["annotations"]

    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    n_done, n_skipped = 0, 0

    for img_info in images:
        image_id = img_info["id"]
        h = img_info["height"]
        w = img_info["width"]

        h, w = int(h), int(w)

        file_name = img_info["file_name"]
        
        if use_filename_stem and "file_name" in img_info and img_info["file_name"]:
            base_w_o_ext = file_name.split(".")[0]
            base = f"{base_w_o_ext.replace("/", "_")}"   
        else:
            base = str(image_id)

        out_path = os.path.join(out_dir, f"{base}.png")

        mask_u8 = anns_to_instance_mask_u8(
            anns_by_img.get(image_id, []),
            h, w,
            overlap=overlap,
            sort_by_area=sort_by_area
        )

        Image.fromarray(mask_u8, mode="L").save(out_path)

        if viz_dir is not None:
            rgb = instance_mask_to_rgb(mask_u8, seed=0)
            viz_path = os.path.join(viz_dir, f"{base}_viz.png")
            Image.fromarray(rgb).save(viz_path)

        n_done += 1

    print(f"Done. Wrote {n_done} masks to: {out_dir}")
    if viz_dir is not None:
        print(f"Also wrote visualizations to: {viz_dir}")
    if n_skipped:
        print(f"Skipped {n_skipped} images due to missing size.")


# -------------------------
# Example CLI-like usage
# -------------------------
if __name__ == "__main__":
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, "data")
    endo2023_dir = path_join(data_dir, "Endoscapes2023")
    annots_dir = path_join(endo2023_dir, "annotations")
    seg_dir = path_join(annots_dir, "Seg50")
    all_annots_dir = path_join(seg_dir, "all_annotation_coco.json")
    
    save_dir = path_join(seg_dir, "masks_png")
    
    
    # Convert all COCO annotations to PNG masks
    coco_json_to_png_masks(
        coco_json_path=all_annots_dir,
        out_dir=save_dir,
        viz_dir="viz_dir",     # set to None to disable
        use_filename_stem=True,      # uses file_name stem if available
        overlap="last",              # or "first"
        sort_by_area=False,          # True can help control overlap priority
        skip_missing_size=False,
    )

