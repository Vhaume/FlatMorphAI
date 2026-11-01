# -*- coding: utf-8 -*-
"""One-time preparation of the Hugging Face files needed by the Mask2Former loader (needs internet once).

mask2former_api.load_model_for_segmentation never downloads weights: it rebuilds the architecture from the cached
config.json of the base model named in the checkpoint (key pretrained_model_name_or_path) and loads the image processor from the cache, then
overwrites every weight with the fine-tuned checkpoint. This script fills models/hf_cache with those two small files
(config and preprocessor, a few kB, no weights), so that the export then runs offline (regen_pipeline.segmentation
sets HF_HOME to models/hf_cache when the folder exists). If the machine already has these files in its own
~/.cache/huggingface, this step is not needed.
Important: without the cached processor the loader falls back on a processor built by hand (shortest edge 800,
longest 1333, ImageNet mean and std), which may differ from the one used for the production masks.

Usage: python segmentation/prepare_hf_cache.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "parameters"))
from parameter_reader import model_path

cache = os.path.join(ROOT, "models", "hf_cache"); os.makedirs(cache, exist_ok=True)
os.environ["HF_HOME"] = cache
from transformers import AutoConfig, AutoImageProcessor

folder = model_path("mask2former_segmentation").rstrip("/\\"); name = os.path.basename(folder)
names = []
try:   # the loader reads the base model name in the checkpoint (key pretrained_model_name_or_path), CPU load
    import torch
    dm = torch.load(os.path.join(folder, name + "_loss_best_loss.pth"), map_location="cpu", weights_only=False)
    names.append(str(dm.get("pretrained_model_name_or_path", "mask2former-swin-tiny-ade-semantic"))); del dm
except Exception as e:
    print("checkpoint not read (%s), default base model used" % e)
names += ["facebook/mask2former-swin-tiny-ade-semantic"]
for n in names:
    for cand in (n, "facebook/" + n if not n.startswith("facebook/") else n):
        try:
            AutoConfig.from_pretrained(cand); AutoImageProcessor.from_pretrained(cand)
            print("cached:", cand)
        except Exception as e:
            print("not available:", cand, "|", str(e).splitlines()[0][:120])
print("Hugging Face cache:", cache)
