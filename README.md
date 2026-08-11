# OpenMP Order Book Parallelization

Parallelizes exchange order-book packet processing with OpenMP, while strictly preserving the sequential semantics required by the assignment (the update stream is inherently order-dependent).

Each packet is a bit-stuffed 49-bit record (`stockID:32 | orderType:1 | orderQty:8 | orderValue:8`); the pipeline de-stuffs, decodes, and folds packets into a running per-stock view.

> **Note on scope**: only `functions.cpp`/`functions.h` were my own submission for this assignment — the course-provided packet feed/driver is intentionally not reproduced here, since it isn't mine to publish. `demo_main.cpp` is a small harness I wrote afterward (not part of the graded submission) purely so this repo builds and runs standalone; it synthesizes a random order book and exercises all three functions below. See the header comment in that file for details, including the bit-stuffing encoder used to generate valid synthetic packets.

## Functions (`functions.cpp`)

- **`updateDisplay`** — the core update loop is kept strictly sequential (each order updates shared book state), but two independent pieces are pulled out and parallelized: packet decoding runs as an upfront `#pragma omp parallel for`, and periodic order-book snapshots are dispatched as OpenMP tasks (`firstprivate` snapshot copy + sort + file write) that run concurrently with the ongoing sequential update stream.
- **`totalAmountTraded`** — embarrassingly parallel: `#pragma omp parallel for reduction(+:total)` over independent per-packet contributions.
- **`printOrderStats`** — each thread accumulates per-stock stats (min sell, max buy, order count/sum) into a thread-local map, merged into the global map under a critical section at the end — avoids per-order lock contention.

## Build & run

```bash
g++ -O3 -std=c++17 -fopenmp functions.cpp demo_main.cpp -o demo
./demo
```

(On macOS, Apple Clang doesn't support `-fopenmp` — use Homebrew GCC, e.g. `g++-15`, or Clang + `libomp`.)

## Results

Measured on a 10-logical-core Apple Silicon machine, speedup vs. the sequential baseline across thread counts {1, 2, 4, 6, 8, 10}:

- `printOrderStats` scales best (~8× at 10 threads) — thread-local aggregation avoids contention almost entirely.
- `totalAmountTraded` follows closely (~6×) as a pure parallel reduction.
- `updateDisplay` shows the most modest speedup (~5×) since its main loop must stay sequential; gains come entirely from parallel pre-decoding and background snapshot tasks.

See `report.pdf` for the full writeup and speedup plot.
