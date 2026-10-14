# STUDY.md: inference optimization, in plain language

Read this before interviews, then open `results/` and be ready to explain **your own** numbers. No benchmark numbers
are quoted here on purpose; every figure you cite should come from your `results.csv` and `profile_summary.md`.

## 1. Why FP16 is faster

Two separate effects:

1. **Tensor cores.** On Ampere, a Tensor Core multiplies small matrix tiles (e.g. 16x8x16) in one instruction, taking
   FP16 inputs and accumulating in FP32 (or FP16). The ordinary FP32 CUDA-core path does one multiply-add per core per
   clock. Convolutions and GEMMs (which is nearly all the FLOPs in ResNet and BERT) map onto tile multiplies, so FP16
   gets a much higher peak FLOP rate than FP32. (Note: the FP32 path in PyTorch can use TF32 tensor cores for convs
   by default; for matmuls it depends on `allow_tf32`. Know which your baseline used.)
2. **Memory bandwidth.** FP16 values are 2 bytes instead of 4. Weights and activations take half the bytes to load and
   store, which matters for **memory-bound** layers (BN, ReLU, add, LayerNorm, softmax, small-batch GEMMs). A laptop
   GPU has a narrow memory bus, so halving bytes helps a lot.

Costs: reduced range/precision (FP16 max ~65504, ~3 decimal digits). Risks: overflow in activations (e.g. in attention
logits), and accumulated error. That is why we check logit differences and top-1 agreement.
*Also relevant:* Tensor Cores are most effective when channel counts are multiples of 8 and tensors are in NHWC
(`channels_last`); that is an optimisation we did **not** apply to config B (possible extension).

## 2. Layer fusion

Eager PyTorch executes one op at a time: conv, then BatchNorm, then ReLU, then add... Each op is a kernel launch that
reads its input from GPU memory and writes the output back. For elementwise ops the arithmetic is trivial and the time
is all memory traffic and launch latency. **Fusion** merges several ops into one kernel so intermediate tensors stay in
registers/shared memory and only the final result is written.
Examples: Conv+BN folded into conv weights (BN is a per-channel affine transform at inference); Conv+Bias+ReLU as a
single kernel epilogue; Add+ReLU; in transformers, bias+GELU, residual+LayerNorm, and fused attention.
Benefits: fewer kernel launches, less memory traffic, less idle time between kernels. You can **see** it in the Nsight
numbers: kernel launches per inference in eager vs TensorRT.

## 3. What TensorRT does

* **Graph optimisation:** parse ONNX, remove no-ops, constant-fold, fold BN into conv, fuse layers (vertical: conv+relu;
  horizontal: parallel convs with the same input), eliminate transposes/concats where possible.
* **Precision selection / calibration:** build with FP16 and/or INT8 allowed; picks per layer what is fastest and
  allowed. INT8 needs scale factors per tensor, found via calibration (section 7).
* **Kernel auto-tuning:** for every layer it times many candidate kernels ("tactics") on *your* GPU, for the shapes in
  your optimization profile, and keeps the fastest. That is why engines are GPU- and TensorRT-version-specific and why
  building takes minutes.
* **Memory planning:** activation tensors with non-overlapping lifetimes share the same memory, so peak memory is much
  lower than "sum of all activations". Workspace size is capped by the builder config.
* **Runtime:** a lean executor with no Python, no autograd, no dispatcher; optional CUDA graphs.

## 4. Compute-bound vs memory-bound

Arithmetic intensity = FLOPs per byte moved. Compare it to the GPU's ratio (peak FLOPs / peak bandwidth), the
"ridge point" of the **roofline model**.
* **Compute-bound** (intensity above ridge): large convs, large GEMMs. Speed is limited by math units -> Tensor Cores,
  INT8, bigger tiles help.
* **Memory-bound** (below ridge): BN, ReLU, add, softmax, LayerNorm, depthwise convs, batch-1 GEMMs (each weight is read
  once and used once). Speed is limited by bandwidth -> fusion, lower precision (fewer bytes), and caching help.
Nsight Compute shows this directly: SM/tensor throughput % vs DRAM throughput % (the optional `profile_ncu.sh` step).
Rule of thumb: whichever % is higher is the limiter; if both are low, you are latency/occupancy-bound.

## 5. What batch size does to utilisation

A GPU needs many independent threads to hide latency. With batch 1 a layer may launch too few thread blocks to fill
all SMs, and each kernel is short enough that launch overhead and CPU-side time matter. Increasing batch size raises
arithmetic intensity (weights are reused across samples), fills the SMs, and amortises launch overhead, so
**throughput rises**, until the GPU is saturated; then it flattens while **latency grows roughly linearly**. Large
batches also cost memory (activations scale with batch), which is the binding limit on a 4 GB card.
Trade-off: online serving cares about latency under an SLO (small batches, dynamic batching with a timeout); offline
jobs care about throughput (largest batch that fits).

## 6. Why p99 matters

Mean/median hide the slow tail, and users and SLOs experience the tail: if a page needs 20 model calls, most users hit
at least one p99 event. Tails come from: clock/power changes, OS scheduling and background processes, the Python GC,
cudnn autotuning on a new shape, allocator cache misses (cudaMalloc), thermal throttling, and shared GPU use (a
laptop GPU also drives the display). With 100-200 samples p99 is basically the 2nd-worst sample: report it, but use
more iterations (1000+) before claiming anything about it.

## 7. How INT8 calibration works

INT8 maps real values to integers with a scale: `q = clamp(round(x / s), -127, 127)`. Weights are quantised per output
channel (their range is known). **Activations** have data-dependent ranges, so we run a small **calibration set**
(representative real inputs, here 32-64 images) through the FP32/FP16 network, record a histogram of every activation
tensor, and choose a clipping threshold per tensor. TensorRT's entropy calibrator (v2) picks the threshold that
minimises KL divergence between the original and quantised distribution (clipping rare outliers gives finer resolution
for the bulk of values). Then scale = threshold / 127. The scales are stored in a calibration cache.
Pitfalls: calibration data must resemble deployment data (noise gives bad scales; this project refuses to calibrate on
noise); too few images give noisy histograms; sensitive layers (first/last, attention softmax) may need to stay in
higher precision. Alternative: Quantisation-Aware Training (the model learns to be robust to quantisation).
INT8 Tensor Cores give higher peak math rate than FP16, and activations are half the bytes of FP16.

## 8. Ten likely interview questions

**1. Why did FP16 speed things up, and when does it not?**
Tensor cores (higher math rate) and half the memory traffic. It helps little when the workload is launch-bound (tiny
batch, small kernels) or dominated by ops that are neither tensor-core nor bandwidth limited; and it can overflow or
lose accuracy, so you validate with output diffs.

**2. What does cudnn.benchmark=True do and when can it hurt?**
cuDNN times its available convolution algorithms for each new input shape and caches the fastest. Good for fixed
shapes; with constantly changing shapes it re-tunes every time (slow first iterations, latency spikes). It can also
make runs non-deterministic.

**3. Why do you warm up before timing?**
First calls include CUDA context/lazy module loading, cuDNN/cuBLAS handle creation, autotuning, allocator growth and
clock ramp-up. Warmup removes these one-time costs from the measurement.

**4. Why synchronize or use CUDA events?**
CUDA launches are asynchronous: wall-clock around `model(x)` measures only launch time. Events (or
`torch.cuda.synchronize()`) measure when the GPU actually finishes.

**5. How is TensorRT faster than PyTorch eager, mechanically?**
Layer fusion and fewer kernel launches, per-layer kernel auto-tuning on the target GPU, lower precision, static memory
planning, and no Python/dispatcher overhead. Evidence: kernels per inference and idle-between-kernels from nsys.

**6. How would you decide between latency and throughput optimisation?**
Start from the SLO. Online: minimise p99 at the request rate (small batch, dynamic batching with a max wait, several
instances). Offline: maximise samples/s at the largest batch that fits memory.

**7. The INT8 model lost accuracy. What do you do?**
Check calibration data quality/size and method (entropy vs minmax/percentile), keep sensitive layers (first conv,
final FC, softmax/LayerNorm paths) in FP16 via per-layer precision, compare per-layer outputs to find the culprit, or
use QAT.

**8. FP16 and INT8 both reduce bytes. Why is INT8 not simply 2x faster again?**
Not everything is quantised (some layers fall back to FP16), reformat/Q-DQ boundaries cost time, memory-bound ops are
limited by bandwidth not math, and small GPUs are often launch/occupancy-bound. Measure, don't assume.

**9. What does it mean if the GPU is idle between kernels?**
The GPU waits for the CPU to launch the next kernel (launch-bound: small kernels, Python overhead). Remedies: bigger
batches, fusion, CUDA graphs, torch.compile (mode "reduce-overhead"), TensorRT.

**10. How do you know whether a kernel is compute- or memory-bound?**
Compute arithmetic intensity (FLOPs/byte) against the GPU's ridge point, or profile with Nsight Compute and compare
SM/tensor-pipe utilisation with DRAM throughput. Whichever is closer to peak is the limiter; both low = latency or
occupancy problem.

**Bonus:** *Why can an engine not be moved to another GPU?* Tactics are tuned on and compiled for a specific
architecture and TensorRT version. *Why does `torch.compile` need warmup?* It compiles lazily on the first call per shape.

## 9. Limitations of these benchmarks

* **Small, shared GPU.** A 4 GB laptop part with a narrow memory bus; conclusions (e.g. how much FP16 helps, where
  batch scaling flattens) do not transfer to A100/H100-class GPUs. It likely also drives your display, so the desktop
  competes for the GPU.
* **Thermal and power throttling.** Laptop GPUs change clocks with temperature and power budget (on battery/"quiet" mode
  especially). Later configs may run at lower clocks than earlier ones. We record `sm_clock_mhz`, `temp_c`, `power_w`
  after each measurement; for fair comparison use AC power, performance mode, a cool start, and repeat runs in a
  different order.
* **Run-to-run noise.** One process, one run per cell: no confidence intervals. p99 from 200 samples is a weak
  estimate. Repeat and compare before drawing small conclusions.
* **Single-stream, fixed shapes, synthetic-ish inputs.** No dynamic batching, no multi-stream concurrency, no
  data-loading/preprocessing in the timing (GPU-resident inputs only), BERT always padded to 128.
* **Accuracy is equivalence, not task accuracy.** Top-1 agreement vs FP32 on a 64-sample set (real images if you
  supply them; noise otherwise); BERT uses a random classification head. A small agreement drop on 64 samples is not
  statistically meaningful.
* **TensorRT engine uses one dynamic-batch profile tuned for batch 16.** Per-batch engines could be faster at 1 and 32.
* **Memory metric is a device-level delta,** including allocator caching in PyTorch; it overstates what a tight
  deployment needs and is not comparable to a `torch.cuda.max_memory_allocated()` number.
* **Profiler overhead.** nsys adds overhead to the profiled iterations; use the profile for *structure* (kernel counts,
  breakdown), the benchmark for *timing*. The nsys idle-time metric spans the whole capture window of 20 back-to-back
  iterations.
* **INT8 depends on your calibration images.** A tiny calibration set gives a coarse result.
* **Software-version dependent.** PyTorch, TensorRT, driver versions all change results; record them (printed at the
  start of each bench run).
