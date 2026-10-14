Speedup vs config A (eager FP32), computed from this machine's `results.csv` (throughput ratio; p50 latency ratio in brackets):

**bert**

| config | bs=1 | bs=4 | bs=16 | bs=32 |
|---|---|---|---|---|
| A | 1.00x (1.00x) | 1.00x (1.00x) | 1.00x (1.00x) | 1.00x (1.00x) |
| B | 1.54x (1.81x) | 2.90x (2.86x) | 3.38x (3.35x) | 7.02x (7.85x) |
| C | 3.61x (3.61x) | 3.77x (3.53x) | 3.69x (3.59x) | 7.59x (8.50x) |
| D | 4.79x (4.82x) | 5.47x (5.53x) | 5.77x (5.75x) | 11.12x (12.45x) |

**resnet50**

| config | bs=1 | bs=4 | bs=16 | bs=32 |
|---|---|---|---|---|
| A | 1.00x (1.00x) | 1.00x (1.00x) | 1.00x (1.00x) | 1.00x (1.00x) |
| B | 1.22x (1.24x) | 1.76x (1.80x) | 2.02x (2.01x) | 1.85x (1.85x) |
| C | 2.19x (1.86x) | 2.56x (2.44x) | 2.85x (2.78x) | 2.60x (2.57x) |
| D | 3.45x (3.17x) | 5.13x (5.15x) | 6.41x (6.43x) | 6.09x (6.11x) |
| E | 5.40x (5.44x) | 10.05x (9.81x) | 13.79x (13.76x) | 13.32x (13.32x) |

Accuracy vs FP32 baseline (64-sample set):

| model | config | max abs logit diff | top-1 agreement % | data |
|---|---|---|---|---|
| bert | A | 0 | 100.0 | builtin sentences (padded to 128) |
| bert | B | 0.001352 | 100.0 | builtin sentences (padded to 128) |
| bert | C | 0.0007929 | 100.0 | builtin sentences (padded to 128) |
| bert | D | 0.00253 | 100.0 | builtin sentences (padded to 128) |
| resnet50 | A | 0 | 100.0 | real:48 images (tiled to 64) |
| resnet50 | B | 0.05047 | 100.0 | real:48 images (tiled to 64) |
| resnet50 | C | 0.05277 | 100.0 | real:48 images (tiled to 64) |
| resnet50 | D | 0.1236 | 96.9 | real:48 images (tiled to 64) |
| resnet50 | E | 3.565 | 84.4 | real:48 images (tiled to 64) |