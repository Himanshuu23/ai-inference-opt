"""ONNX export + TensorRT engine building through the TensorRT Python API (no trtexec needed).

One engine per (model, precision) with a dynamic-batch optimization profile min=1 / opt=16 / max=32.
Kernel tactics are tuned for the opt shape (batch 16); other batch sizes reuse the same engine.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .common import ENGINES, TRT_MAX_BATCH, TRT_MIN_BATCH, TRT_OPT_BATCH
from .models import Bundle, calibration_images, load_model


def onnx_path(b: Bundle) -> Path:
    tag = b.name if b.name == "resnet50" else b.bert_id.replace("/", "_")
    return ENGINES / f"{tag}.onnx"


def engine_path(b: Bundle, precision: str) -> Path:
    tag = b.name if b.name == "resnet50" else b.bert_id.replace("/", "_")
    return ENGINES / f"{tag}_{precision}.engine"


def export_onnx(b: Bundle, force: bool = False) -> Path:
    path = onnx_path(b)
    if path.exists() and not force:
        return path
    ENGINES.mkdir(exist_ok=True)
    print(f"[onnx] exporting {b.name} -> {path}")
    model = load_model(b.name, b.bert_id, eager_attn=True).eval()  # eager attention exports cleanly
    dummy = []
    for n in b.input_names:
        shape = (2,) + tuple(b.input_specs[n])
        dt = b.onnx_dtypes[n]
        dummy.append(torch.randn(shape) if dt.is_floating_point else torch.randint(0, 1000, shape, dtype=dt))
    kw = dict(input_names=b.input_names, output_names=["logits"], opset_version=17, do_constant_folding=True,
              dynamic_axes={n: {0: "batch"} for n in b.input_names + ["logits"]})
    with torch.no_grad():
        try:
            torch.onnx.export(model, tuple(dummy), str(path), dynamo=False, **kw)  # torch>=2.5: force legacy exporter
        except TypeError:
            torch.onnx.export(model, tuple(dummy), str(path), **kw)
    import onnx
    onnx.checker.check_model(str(path))
    print(f"[onnx] ok ({path.stat().st_size / 2**20:.1f} MiB)")
    return path


def _make_calibrator(data: torch.Tensor, batch: int, cache_file: Path):
    import tensorrt as trt

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.data, self.batch, self.i, self.cache_file, self.buf = data, batch, 0, cache_file, None

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.i + self.batch > len(self.data):
                return None
            self.buf = self.data[self.i:self.i + self.batch].cuda().contiguous()
            self.i += self.batch
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            return self.cache_file.read_bytes() if self.cache_file.exists() else None

        def write_calibration_cache(self, cache):
            self.cache_file.write_bytes(cache)

    return EntropyCalibrator()


def build_engine(b: Bundle, precision: str, workspace_mb: int = 1024) -> Path:
    import tensorrt as trt
    assert precision in ("fp16", "int8")
    onnx_file = export_onnx(b)
    out = engine_path(b, precision)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") and int(trt.__version__.split(".")[0]) < 10:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_file.read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)  # INT8 builds also allow FP16 fallback for layers that stay unquantized
    profile = builder.create_optimization_profile()
    for n in b.input_names:
        s = tuple(b.input_specs[n])
        profile.set_shape(n, (TRT_MIN_BATCH,) + s, (TRT_OPT_BATCH,) + s, (TRT_MAX_BATCH,) + s)
    config.add_optimization_profile(profile)

    if precision == "int8":
        if b.name != "resnet50":
            raise RuntimeError("INT8 is only implemented for resnet50")
        imgs = calibration_images(64)
        if imgs is None:
            raise RuntimeError("TODO: INT8 needs >=32 real images in data/images (calibration on noise is meaningless)")
        n_use = len(imgs) // TRT_OPT_BATCH * TRT_OPT_BATCH
        cache = ENGINES / "resnet50_int8.calib"
        if cache.exists():
            cache.unlink()  # always recalibrate so the cache matches the current image set
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_calibration_profile(profile)
        config.int8_calibrator = _make_calibrator(imgs[:n_use], TRT_OPT_BATCH, cache)
        print(f"[trt] INT8 calibration on {n_use} real images, entropy calibrator v2")

    print(f"[trt] building {precision} engine for {b.name} (this can take minutes on a small GPU)...")
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build returned None (see log above; maybe out of memory -> lower workspace_mb)")
    ENGINES.mkdir(exist_ok=True)
    out.write_bytes(bytes(blob))
    print(f"[trt] built {out.name} in {time.time() - t0:.0f}s ({out.stat().st_size / 2**20:.1f} MiB)")
    return out


def ensure_engine(b: Bundle, precision: str, rebuild: bool = False) -> Path:
    p = engine_path(b, precision)
    if rebuild or not p.exists():
        build_engine(b, precision)
    return p
