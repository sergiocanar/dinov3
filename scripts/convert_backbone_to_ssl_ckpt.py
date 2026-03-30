"""
Convert a raw DINOv3 backbone .pth (flat state dict) into the format expected
by init_fsdp_model_from_checkpoint, which reads ckpt["teacher"] and expects
keys prefixed with "backbone.".

Usage:
    python scripts/convert_backbone_to_ssl_ckpt.py \
        --input  weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --output weights/dinov3_vitl16_pretrain_ssl_format.pth
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    backbone_sd = torch.load(args.input, map_location="cpu")
    assert isinstance(backbone_sd, dict), "Expected a flat state dict"

    wrapped = {"backbone." + k: v for k, v in backbone_sd.items()}
    torch.save({"teacher": wrapped}, args.output)

    print(f"Saved {len(wrapped)} keys → {args.output}")
    print("Sample keys:", list(wrapped.keys())[:4])


if __name__ == "__main__":
    main()
