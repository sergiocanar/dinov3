import os 
from os.path import join as path_join

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