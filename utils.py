import os 
from os.path import join as path_join
import torch

import json

def load_json(json_path: str)->dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: dict, save_path: str)->None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def join_coco_json(json_lt: list[str], save_dir:str)->None:
    
    for i, json_path in enumerate(json_lt):
        data = load_json(json_path)
        if i == 0:
            combined_data = data
        else:
            combined_data['images'].extend(data['images'])
            combined_data['annotations'].extend(data['annotations'])

    save_json(combined_data, save_dir)
    print(f"Combined {len(json_lt)} COCO JSON files into {save_dir}")

def load_txt(path: str) -> str:
    '''Load a text file and return its contents as a list.'''
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip())
        
    return data

def cvs_dino_collator(batch):
    # batch items: (img, label, video_name, frame_id, metadata)
    imgs = []
    labels = []
    video_names = []
    frame_ids = []

    for img, label, video_name, frame_id, metadata in batch:

        imgs.append(img)
        labels.append(label)
        video_names.append(video_name)
        frame_ids.append(int(frame_id))

    pixel_values = torch.stack(imgs, dim=0)                 # (B,3,224,224)
    labels = torch.stack(labels, dim=0).float()             # (B,3) for BCEWithLogitsLoss

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        # optional but handy; keep remove_unused_columns=False
        "video_name": video_names,
        "frame_id": torch.tensor(frame_ids, dtype=torch.long),
    }

def cvs_data_collator(batch, feature_extractor):
    images = [img for img, label, video_name, frame_id, metadata in batch]
    labels = torch.stack([label for img, label, video_name, frame_id, metadata in batch], dim=0)   # [B, 3]

    enc = feature_extractor(images, return_tensors="pt")
    pixel_values = enc["pixel_values"]  # [B, 3, 224, 224]

    # Return only what the model forward expects:
    return {"pixel_values": pixel_values, "labels": labels}
