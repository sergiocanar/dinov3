import os
import csv
from os.path import join as path_join

from utils import load_json, load_txt


def normalize_vid_id(s: str) -> str:
    """
    Normalizes video identifiers coming from split txt files.
    Accepts:
      - 'video_004'
      - '4'
      - '004'
    Returns:
      - 'video_004'
    """
    vid_num = int(float(s))
    vid_name = f"video_{vid_num:03d}"
    return vid_name


def get_video_id_from_filename(file_name: str) -> str:
    """
    Extracts video id from COCO image 'file_name'.
    Expected examples:
      'video_004/000123.jpg' -> 'video_004'
      'frames/video_004/000123.jpg' -> 'video_004'
    """
    parts = file_name.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("video_"):
            return p
    # If you use another pattern, adapt here
    raise ValueError(f"Could not parse video id from file_name='{file_name}'")


def extract_3_labels(ds):
    """
    ds is expected to be list/tuple length 3.
    """
    if isinstance(ds, (list, tuple)) and len(ds) == 3:
        return int(ds[0]), int(ds[1]), int(ds[2])
    raise TypeError(f"Expected ds to be list/tuple of len 3. Got: {type(ds)} with value={ds}")


def write_csv(rows, out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "c1", "c2", "c3"])
        w.writerows(rows)
    print(f"[OK] Wrote: {out_csv_path} ({len(rows)} rows)")


def coco_to_cvs_csv_splits(
    coco_json_path: str,
    images_root: str,
    out_dir: str,
    train_vids_txt: str,
    val_vids_txt: str,
    test_vids_txt: str,
    ds_key: str = "ds",
):
    coco = load_json(coco_json_path)

    # Load split sets
    train_set = set(map(normalize_vid_id, load_txt(train_vids_txt)))
    val_set   = set(map(normalize_vid_id, load_txt(val_vids_txt)))
    test_set  = set(map(normalize_vid_id, load_txt(test_vids_txt)))

    # Optional sanity: ensure no overlap
    inter_tv = train_set & val_set
    inter_tt = train_set & test_set
    inter_vt = val_set & test_set
    if inter_tv or inter_tt or inter_vt:
        print("[WARN] Video split overlap detected!")
        print(" train∩val :", sorted(inter_tv)[:10])
        print(" train∩test:", sorted(inter_tt)[:10])
        print(" val∩test  :", sorted(inter_vt)[:10])

    rows_all, rows_train, rows_val, rows_test = [], [], [], []

    for img in coco["images"]:
        if ds_key not in img:
            raise KeyError(f"Missing '{ds_key}' in image entry: id={img.get('id')} keys={list(img.keys())}")

        file_name = img["file_name"]
        vid_id = get_video_id_from_filename(file_name)

        y1, y2, y3 = extract_3_labels(img[ds_key])

        image_path = path_join(images_root, file_name)
        row = [image_path, y1, y2, y3]

        rows_all.append(row)
        if vid_id in train_set:
            rows_train.append(row)
        elif vid_id in val_set:
            rows_val.append(row)
        elif vid_id in test_set:
            rows_test.append(row)
        else:
            # video not listed in any split file
            # keep in all.csv but not in train/val/test
            pass

    # Write outputs
    write_csv(rows_all,  path_join(out_dir, "all.csv"))
    write_csv(rows_train, path_join(out_dir, "train.csv"))
    write_csv(rows_val,   path_join(out_dir, "val.csv"))
    write_csv(rows_test,  path_join(out_dir, "test.csv"))


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = path_join(this_dir, "data")
    endoscapes_dir = path_join(data_dir, "Endoscapes2023")
    annots_dir = path_join(endoscapes_dir, "annotations", "CVS201")

    coco_json_path = path_join(annots_dir, "all_annotation_coco.json")
    images_root = path_join(endoscapes_dir, "frames")  # prefix for file_name
    out_dir = annots_dir  # where csv files will be saved

    train_vids_txt = path_join(data_dir, "train_vids.txt")
    val_vids_txt   = path_join(data_dir, "val_vids.txt")
    test_vids_txt  = path_join(data_dir, "test_vids.txt")  # you said test.txt; rename or set accordingly

    coco_to_cvs_csv_splits(
        coco_json_path=coco_json_path,
        images_root=images_root,
        out_dir=out_dir,
        train_vids_txt=train_vids_txt,
        val_vids_txt=val_vids_txt,
        test_vids_txt=test_vids_txt,
        ds_key="ds",
    )
