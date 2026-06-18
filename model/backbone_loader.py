import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import open_clip
from transformers import CLIPModel, CLIPTokenizer


BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
RADCLIP_BASE_MODEL_ID = "openai/clip-vit-large-patch14"


class OpenClipTokenizerWrapper:
    def __init__(self, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(self, texts):
        return self.tokenizer(texts, context_length=self.context_length)


class HFTokenizerWrapper:
    def __init__(self, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )


class RadCLIPWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        output_dim = self.model.visual_projection.out_features
        self.visual = SimpleNamespace(output_dim=output_dim)
        self.logit_scale = self.model.logit_scale
        self.vision_model = self.model.vision_model
        self.visual_projection = self.model.visual_projection

    def encode_image(self, image):
        vision_outputs = self.vision_model(pixel_values=image)
        pooled = vision_outputs.pooler_output
        projected = self.visual_projection(pooled)
        return projected

    def encode_text(self, text_inputs):
        return self.model.get_text_features(**text_inputs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _load_radclip_state_dict(backbone_ckpt):
    checkpoint = torch.load(backbone_ckpt, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def build_backbone(model_cfg, device):
    backbone_name = getattr(model_cfg, "backbone_name", "biomedclip").lower()

    if backbone_name == "biomedclip":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL_ID,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        )
        tokenizer = OpenClipTokenizerWrapper(
            open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID),
            context_length=256,
        )
        context_length = 256
        clip_model.to(device).eval()
        return clip_model, tokenizer, context_length

    if backbone_name == "radclip":
        if not getattr(model_cfg, "backbone_ckpt", ""):
            raise ValueError("model.backbone_ckpt must be set when backbone_name='radclip'.")
        hf_model = CLIPModel.from_pretrained(RADCLIP_BASE_MODEL_ID)
        state_dict = _load_radclip_state_dict(model_cfg.backbone_ckpt)
        missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[RadCLIP] Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"[RadCLIP] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
        clip_model = RadCLIPWrapper(hf_model)
        tokenizer = HFTokenizerWrapper(
            CLIPTokenizer.from_pretrained(RADCLIP_BASE_MODEL_ID),
            context_length=77,
        )
        context_length = 77
        clip_model.to(device).eval()
        return clip_model, tokenizer, context_length

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")
