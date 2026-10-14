"""Pre-build ONNX + TensorRT engines:  python -m src.build_engines --model resnet50 --precisions fp16 int8"""
import argparse

from .models import load_bundle
from .trt_utils import ensure_engine

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50", choices=["resnet50", "bert"])
ap.add_argument("--precisions", nargs="+", default=["fp16"], choices=["fp16", "int8"])
ap.add_argument("--bert-model", default="bert-base-uncased")
ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()
b = load_bundle(a.model, a.bert_model)
for p in a.precisions:
    print(ensure_engine(b, p, a.rebuild))
