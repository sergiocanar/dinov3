import os
from os.path import join as path_join

import urllib
import pickle
from dinov3.hub.backbones import dinov3_vit7b16


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import ConnectionPatch

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF


if __name__ == "__main__":
    
    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

    if os.getenv("DINOV3_LOCATION") is not None:
        print("Using DINOv3 location from environment variable DINOV3_LOCATION")
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

    print(f"DINOv3 location set to {DINOV3_LOCATION}")
    
    # examples of available DINOv3 models:
    MODEL_DINOV3_VITS = "dinov3_vits16"
    MODEL_DINOV3_VITSP = "dinov3_vits16plus"
    MODEL_DINOV3_VITB = "dinov3_vitb16"
    MODEL_DINOV3_VITL = "dinov3_vitl16"
    MODEL_DINOV3_VITHP = "dinov3_vith16plus"
    MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

    # we take DINOv3 ViT-B
    MODEL_NAME = MODEL_DINOV3_VITB

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=MODEL_NAME,
        weights="weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
 
    model.cuda()
    
