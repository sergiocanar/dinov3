import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


    
class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        # expose Linear-like API expected by dinov3 attention
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class PatchTransformerHead(nn.Module):
    def __init__(self, d_model, num_labels=3, n_tokens=256, depth=4, nheads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens, d_model))  # assumes fixed N
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=d_model * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_labels)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, patch_tokens):  # (B, N, D)
        B, N, D = patch_tokens.shape
        cls = self.cls.expand(B, -1, -1)                 # (B,1,D)
        x = torch.cat([cls, patch_tokens], dim=1)        # (B,1+N,D)

        # If N differs from init, interpolate pos embeddings
        if self.pos.shape[1] != x.shape[1]:
            pos = F.interpolate(
                self.pos.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos

        x = x + pos
        x = self.enc(x)
        x = self.norm(x[:, 0])                           # CLS
        return self.fc(x)                                # (B,3)


class CVSHeadMLP(nn.Module):
    """
    Simple multi-label head:
    concat CLS tokens from multiple layers -> MLP -> logits (3)
    """
    def __init__(self, in_dim, num_labels=3, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):  # x: (B, in_dim)
        return self.net(x)

class SurgicalDINOForCVS(nn.Module):
    def __init__(
        self,
        backbone_size="base",
        r=4,
        lora_layer=None,
        num_labels=3,
        use_layers="4",
        head_type="mlp",            # "mlp" or "transformer"
        tf_depth=4,
        tf_heads=8,
        tf_mlp_ratio=4,
        tf_dropout=0.1,
        img_size=224,               # needed to know number of patch tokens
        patch_size=16,              # dinov3 is ViT-16
        weights_dir=None
        ):
        super().__init__()
        
        #Model size
        archs = {"small": "vits16", 
                 "small_plus": "vits16plus",
                 "base": "vitb16", 
                 "large": "vitl16", 
                 "huge_plus": "vith16plus",
                 "giant": "vitg16"
        }
        
        MODEL_NAME = f"dinov3_{archs[backbone_size]}"
        source = "local" 
        dinov3_dir = "/home/scanar/endovis/models/dinov3/"

        dinov3 = torch.hub.load(
            repo_or_dir=dinov3_dir,
            model=MODEL_NAME,
            source=source,
            weights=weights_dir
        )
        
        #Layer size
        inter = {
            "small": [2, 5, 8, 11],
            "base":  [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        
        #Embedding dimension 
        dims = {"small": 384, 
                "base": 768, 
                "large": 1024, 
                "giant": 1536}
        
        #Dinov2 parameters
        self.backbone_size = backbone_size
        self.backbone_name = f"dinov3_{archs[backbone_size]}"
        self.layers = inter[backbone_size]
        self.dim = dims[backbone_size]
        self.use_layers = use_layers

        # freeze backbone
        for p in dinov3.parameters():
            p.requires_grad = False

        # lora layers
        if lora_layer is None:
            lora_layer = list(range(len(dinov3.blocks)))
        self.lora_layer = lora_layer
        
        self.w_As, self.w_Bs = [], []

        for i, blk in enumerate(dinov3.blocks):
            if i not in self.lora_layer:
                continue
            qkv = blk.attn.qkv
            d = qkv.in_features

            w_a_q = nn.Linear(d, r, bias=False)
            w_b_q = nn.Linear(r, d, bias=False)
            w_a_v = nn.Linear(d, r, bias=False)
            w_b_v = nn.Linear(r, d, bias=False)

            self.w_As += [w_a_q, w_a_v]
            self.w_Bs += [w_b_q, w_b_v]

            blk.attn.qkv = _LoRA_qkv(qkv, w_a_q, w_b_q, w_a_v, w_b_v)

        self._reset_lora()
        self.dinov3 = dinov3

        # ---- head selection ----
        if head_type == "mlp":
            # MLP uses CLS tokens
            head_in = (4 * self.dim) if self.use_layers == "4" else self.dim
            self.head = CVSHeadMLP(head_in, num_labels=num_labels)

        elif head_type == "transformer":
            # Transformer uses patch tokens (recommended)
            # number of patch tokens for ViT-14: (H/14)*(W/14)
            H = img_size // patch_size
            W = img_size // patch_size
            n_tokens = H * W

            self.head = PatchTransformerHead(
                d_model=self.dim,
                num_labels=num_labels,
                n_tokens=n_tokens,
                depth=tf_depth,
                nheads=tf_heads,
                mlp_ratio=tf_mlp_ratio,
                dropout=tf_dropout,
            )
        else:
            raise ValueError(f"Unknown head_type={head_type}")
        
        self.head_type = head_type
        pos_weight = torch.tensor([6.80, 3.04, 5.44], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _reset_lora(self):
        for wA in self.w_As:
            nn.init.kaiming_uniform_(wA.weight, a=math.sqrt(5))
        for wB in self.w_Bs:
            nn.init.zeros_(wB.weight)
            
    def forward(self, pixel_values, labels=None, **kwargs):
        feats = self.dinov3.get_intermediate_layers(
            pixel_values,
            n=self.layers if self.use_layers == "4" else [self.layers[-1]],
            reshape=False,
            return_class_token=True,
            norm=False,
        )
        # feats: list of tuples [(patch_tokens, cls_token), ...]

        if self.head_type == "mlp":
            # CLS-based
            if self.use_layers == "4":
                cls = torch.cat([f[1] for f in feats], dim=-1)  # (B, 4*dim)
            else:
                cls = feats[0][1]  # (B, dim)

            logits = self.head(cls)
            emb = cls

        else:
            # Transformer-based (patch tokens)
            # Use the LAST layer patch tokens (most standard).
            patch = feats[-1][0]  # (B, N, dim)
            logits = self.head(patch)
            emb = patch.mean(dim=1)  # lightweight embedding for logging

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        return {"loss": loss, "logits": logits, "probs": probs, "emb": emb}

    def trainable_parameters(self):
        # convenience: only LoRA + head
        return list(self.head.parameters()) + [p for p in self.parameters() if p.requires_grad]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    surgicaldino = SurgicalDINOForCVS(
        backbone_size="base",
        head_type="transformer",      # or "transformer"
        use_layers="4",
        weights_dir="/home/scanar/endovis/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ).to(device)

    surgicaldino.train()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (2, 3), device=device)  # multi-label (B,3) in {0,1}

    out = surgicaldino(pixel_values=x, labels=y)
    print("logits:", out["logits"].shape)  # (B,3)
    print("probs:", out["probs"].shape)    # (B,3)
    print("loss:", out["loss"])
    