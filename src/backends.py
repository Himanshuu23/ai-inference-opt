"""Runners for the optimisation levels. Interface:

    r = make_runner(cfg, bundle, example_inputs)   # example_inputs: canonical CPU dict
    r.bind(inputs)    # move/convert inputs to the GPU (NOT timed)
    out = r.run()     # one inference, async on the current CUDA stream (timed)
    r.close()         # free GPU memory
"""
from __future__ import annotations

import copy
import gc

import torch

from .models import Bundle
from .trt_utils import ensure_engine


class Runner:
    note = ""

    def bind(self, inputs: dict): ...
    def run(self): ...
    def close(self): ...


class TorchRunner(Runner):
    def __init__(self, b: Bundle, half: bool, cudnn_benchmark: bool = False):
        self.b, self.half = b, half
        self.prev_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if half:
            self.model = copy.deepcopy(b.fp32).half().cuda().eval()
        else:
            self.model = b.fp32.cuda().eval()
        self.args = None

    def bind(self, inputs):
        args = []
        for n in self.b.input_names:
            t = inputs[n].cuda(non_blocking=True)
            if self.half and t.is_floating_point():
                t = t.half()
            args.append(t.contiguous())
        self.args = args
        torch.cuda.synchronize()

    def run(self):
        with torch.inference_mode():
            return self.model(*self.args)

    def close(self):
        torch.backends.cudnn.benchmark = self.prev_bench
        if not self.half:
            self.b.fp32.cpu()  # keep the shared FP32 weights on the CPU between configs
        self.model = None
        self.args = None
        gc.collect()
        torch.cuda.empty_cache()


class CompiledRunner(TorchRunner):
    """torch.compile (Inductor) on the FP16 model; falls back to TorchScript trace+freeze if compile fails
    (typical on Windows where Triton is unavailable)."""

    def __init__(self, b: Bundle, example: dict):
        super().__init__(b, half=True, cudnn_benchmark=True)
        base = self.model
        self.bind(example)
        try:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 32
            self.model = torch.compile(base)
            self.run()
            torch.cuda.synchronize()
            self.note = "torch.compile (inductor, default mode)"
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:120]
            with torch.no_grad():
                traced = torch.jit.trace(base, tuple(self.args), check_trace=False)
                self.model = torch.jit.freeze(traced.eval())
                for _ in range(3):  # run profiling executions so the TorchScript optimiser kicks in
                    self.model(*self.args)
            torch.cuda.synchronize()
            self.note = f"TorchScript trace+freeze (torch.compile failed: {msg})"


class TRTRunner(Runner):
    def __init__(self, path, b: Bundle):
        import numpy as np
        import tensorrt as trt
        from pathlib import Path
        self.trt, self.b = trt, b
        self.np2torch = {np.float32: torch.float32, np.float16: torch.float16,
                         np.int32: torch.int32, np.int64: torch.int64, np.int8: torch.int8}
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError("could not deserialize engine (built with a different TensorRT/GPU? use --rebuild)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out_name = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.bufs, self.out = [], None

    def _tdtype(self, name):
        return self.np2torch[self.trt.nptype(self.engine.get_tensor_dtype(name))]

    def bind(self, inputs):
        self.bufs = []
        for n in self.in_names:
            t = inputs[n].to("cuda").to(self._tdtype(n)).contiguous()
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
            self.bufs.append(t)
        shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        self.out = torch.empty(shape, dtype=self._tdtype(self.out_name), device="cuda")
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        torch.cuda.synchronize()

    def run(self):
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed (possible out of memory)")
        return self.out

    def close(self):
        self.bufs, self.out, self.ctx, self.engine = [], None, None, None
        gc.collect()
        torch.cuda.empty_cache()


def make_runner(cfg: str, b: Bundle, example: dict, rebuild: bool = False) -> Runner:
    if cfg == "A":
        return TorchRunner(b, half=False, cudnn_benchmark=False)
    if cfg == "B":
        return TorchRunner(b, half=True, cudnn_benchmark=True)
    if cfg == "C":
        return CompiledRunner(b, example)
    if cfg == "D":
        return TRTRunner(ensure_engine(b, "fp16", rebuild), b)
    if cfg == "E":
        if b.name != "resnet50":
            raise RuntimeError("E (INT8) is ResNet-50 only")
        return TRTRunner(ensure_engine(b, "int8", rebuild), b)
    raise ValueError(cfg)
