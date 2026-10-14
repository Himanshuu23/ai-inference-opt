"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ENGINES = ROOT / "engines"
DATA = ROOT / "data"

BATCH_SIZES = [1, 4, 16, 32]
SEQ_LEN = 128
# TensorRT optimization profile (dynamic batch): min / opt / max
TRT_MIN_BATCH, TRT_OPT_BATCH, TRT_MAX_BATCH = 1, 16, 32

CONFIGS = {
    "A": "PyTorch eager FP32",
    "B": "PyTorch FP16 + cudnn.benchmark",
    "C": "torch.compile FP16",
    "D": "TensorRT FP16",
    "E": "TensorRT INT8 (ResNet-50 only)",
}
