"""Model loading + deterministic input generation for ResNet-50 and a small BERT.

Canonical inputs are a dict name -> CPU tensor:
  resnet50: {"input": float32 [N,3,224,224]}
  bert:     {"input_ids": int64 [N,128], "attention_mask": int64 [N,128]}
Each runner converts to whatever dtype it needs (outside the timed region).

Accuracy note: top-1 agreement is measured against the FP32 baseline's own predictions,
not against ImageNet ground-truth labels (there are no labels in this project).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .common import DATA, SEQ_LEN

SENTENCES = [
    "The new graphics card handled the workload with surprisingly little power.",
    "Batching requests together usually improves throughput but hurts tail latency.",
    "She walked along the river at dawn, listening to the city wake up.",
    "Quantization trades a small amount of accuracy for a large reduction in memory traffic.",
    "Although the experiment failed twice, the team learned something valuable each time.",
    "Tensor cores multiply small matrices in half precision and accumulate in higher precision.",
    "The restaurant was crowded, but the service was quick and the food was excellent.",
    "Profilers show where the time goes; they do not tell you what to change.",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BertWrap(nn.Module):
    """Takes ids/mask (int32 or int64), returns logits only. Casting inside keeps the ONNX inputs int32,
    which TensorRT handles natively."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids.long(), attention_mask=attention_mask.long(), return_dict=False)[0]


@dataclass
class Bundle:
    name: str                      # "resnet50" | "bert"
    bert_id: str
    fp32: nn.Module                # lives on CPU; runners move/copy it to the GPU as needed
    input_names: list
    input_specs: dict              # name -> per-sample shape (no batch dim)
    onnx_dtypes: dict              # name -> torch dtype used for ONNX export
    _cache: dict = field(default_factory=dict)
    data_source: str = ""


def load_model(name: str, bert_id: str, eager_attn: bool = False) -> nn.Module:
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    if name == "bert":
        from transformers import AutoModelForSequenceClassification
        torch.manual_seed(0)  # the classifier head is randomly initialised -> keep it identical across loads
        kw = {"attn_implementation": "eager"} if eager_attn else {}
        m = AutoModelForSequenceClassification.from_pretrained(bert_id, num_labels=2, **kw).eval()
        return BertWrap(m).eval()
    raise ValueError(name)


def load_bundle(name: str, bert_id: str = "bert-base-uncased") -> Bundle:
    fp32 = load_model(name, bert_id)
    if name == "resnet50":
        return Bundle(name, bert_id, fp32, ["input"], {"input": (3, 224, 224)}, {"input": torch.float32})
    return Bundle(name, bert_id, fp32, ["input_ids", "attention_mask"],
                  {"input_ids": (SEQ_LEN,), "attention_mask": (SEQ_LEN,)},
                  {"input_ids": torch.int32, "attention_mask": torch.int32})


def _real_images(max_n: int):
    files = sorted(p for p in (DATA / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)[:max_n]
    if not files:
        return None
    from PIL import Image
    from torchvision import transforms as T
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in files])


def calibration_images(max_n: int = 64):
    """Real images only (INT8 calibration on noise is meaningless). Returns tensor or None."""
    imgs = _real_images(max_n)
    return imgs if imgs is not None and len(imgs) >= 32 else None


def get_inputs(b: Bundle, n: int) -> dict:
    """n samples, deterministic. Real images (data/images) if present, else seeded noise; tiled if fewer than n."""
    if n in b._cache:
        return b._cache[n]
    if b.name == "resnet50":
        real = _real_images(64)
        if real is not None and len(real) >= 8:
            idx = torch.arange(n) % len(real)
            x = real[idx].contiguous()
            b.data_source = f"real:{len(real)} images (tiled to {n})" if len(real) < n else f"real:{len(real)} images"
        else:
            g = torch.Generator().manual_seed(0)
            x = torch.randn(n, 3, 224, 224, generator=g)
            b.data_source = "synthetic-noise (TODO: put >=32 JPEGs in data/images for meaningful top-1)"
        out = {"input": x}
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(b.bert_id)
        texts = [f"{SENTENCES[i % len(SENTENCES)]} (sample {i})" for i in range(n)]
        enc = tok(texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
        out = {"input_ids": enc["input_ids"].long(), "attention_mask": enc["attention_mask"].long()}
        b.data_source = "builtin sentences (padded to 128)"
    b._cache[n] = out
    return out


def take(inputs: dict, n: int, start: int = 0) -> dict:
    return {k: v[start:start + n] for k, v in inputs.items()}
