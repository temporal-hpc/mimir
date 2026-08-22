# Session handoff — RR benchmark paper (remote-rendering)

Context carrier for continuing on another machine. The assistant's persistent memory lives
under `~/.claude/...` and does **not** travel with a `git pull`, so the durable facts are
copied here. Branch: `feature/remote-rendering`.

## Goal
GPU performance-benchmark paper for mimir's remote-rendering pipeline across
**A100 (80GB), H200, RTX PRO 6000 Blackwell (96GB), B300 (268GB)**, in three rendering modes:
**path tracing (`pt`)**, **`phong`** (the impostor path), and **unshaded / `--render-path flat`
(`raster`)**.

## What was done this session (all committed on feature/remote-rendering)

1. **Native-PT memory pre-flight now accounts for the BVH** (`samples/remote-rendering/rr-server.cu`).
   The old pre-flight sized native (no-`--lod`) path tracing at ~40 B/particle
   (positions 12 + ids 4 + AABBs 24, +4 cluster id) and **ignored the BLAS acceleration
   structure**, so it falsely OK'd huge native-PT scenes that then aborted mid-build on a
   Vulkan OOM (or, on the RT-less B300, a `DEVICE_LOST` GPU fault). Now it adds
   **+104 B/particle** for the BVH when `pt_no_lod`. Verified on an L40:
   - 100M native PT → estimate **14.4 GB** vs real post-setup **14.2 GB** (near-exact).
   - 500M native PT → **rejected up front** (`need 72 GB > 47 GB free`) instead of OOM-ing.

2. **Real driver-queried BVH size logged at setup** (`lib/src/raytracing.cpp`,
   `createDynamicBlasChunks`). Example (L40, 100M):
   `Path tracing: BVH acceleration structure 10.55 GB (BLAS storage 3.35 GB + build scratch 7.20 GB, 1 chunk(s))`.

3. **Batch plot driver** (`research/scripts/plot_grid.py`) — imports `plot_benchmark.py` and
   reuses its panels; emits **one 2×3 grid per rendering mode** (rows = Throughput / Timings,
   columns = **100M / 1G / MAX**), GPUs overlaid per cell with stable colors. The **MAX column
   legend names each GPU's N** (A100·5G, RTX PRO 6000·6G, H200·9G, B300·16G).

## How to regenerate the figures
```
python3 research/scripts/plot_grid.py                 # -> research/plots/benchmark-grid-{pt,phong,raster}.pdf
python3 research/scripts/plot_grid.py --format png --logy   # PNG, log-scale timings rows
```
Needs pandas + matplotlib. `research/plots/` is gitignored (regenerable from `research/data/`).

## How to rebuild after the C++ changes (needed on the pods/next machine)
```
./mimir-build-from-change.sh                                  # lib (raytracing.cpp)
./samples-build-from-change.sh --sample remote-rendering      # rr-server.cu
```
Toolchain in the container used: cmake 3.28, nvcc 13.0 (CUDA 13). On HPC nodes `cmake`/`nvcc`
come from `module load` — not in a bare non-interactive shell.

## Durable facts / gotchas (mirrored from assistant memory — these do NOT travel with git)

- **Native PT at 1B is infeasible everywhere.** B300 has **0 RT cores** → software-BVH build+trace
  of 1B AABB spheres runs as one giant driver-emulated compute submission (~63s) then GPU-faults
  (`ERROR_DEVICE_LOST`, surfaces at the next `vkQueueSubmit` = the frame-readback `immediateSubmit`,
  not the real fault site). RTX PRO 6000 (96GB) OOMs at ~105 GB. **The LOD-comparison figure was
  captured at 100M native PT on the RTX PRO 6000.** BVH cost measured ~104 B/particle (single chunk);
  scratch is sized for the largest single chunk (~5.4e8 prims) and not summed across chunks, so
  per-particle cost drops a bit once N exceeds one chunk.
- **RunPod Vulkan fix:** RunPod's `nvidia_icd.json` `library_path` points at `libGLX_nvidia.so.0`
  (returns NULL `vkCreateInstance` → falls back to llvmpipe → rr-server aborts). Fix: add a libEGL
  ICD manifest —
  `echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}' > /etc/vulkan/icd.d/nvidia_egl_icd.json`.
  Diagnose with `vulkaninfo --summary | grep deviceType` (want DISCRETE_GPU, not llvmpipe).
  Container recipe: base image ≤ ubuntu24.04 (glibc) + `libx11-6 libxext6` + the libEGL manifest.
  `/dev/dri` is NOT required for headless NVIDIA Vulkan.
- **Driver 570.x** caps high-N RR runs with a Vulkan OOM (driver bug) — prefer 580+ hosts for
  large-N columns. Driver version is host-dictated, unchangeable inside a container.
- **`--pause-at <step>`** (already on origin) freezes the sim at a step so the same seed-deterministic
  state can be captured under different LOD settings for equal side-by-side screenshots.

## Open / possible next steps
- Assemble the three per-mode grids into the paper (LaTeX subfigures or the single grids as-is).
- The `+104 B/particle` BVH constant was measured on an L40 / CUDA 13; AABB-BVH sizing varies by
  driver/GPU. It is deliberately conservative (errs toward rejecting) and the exact size is always
  logged at setup, so it self-documents. Revisit only if a real run is wrongly rejected.
- Data is in `research/data/` (36 CSVs, naming: `mimir-<date>-rr-client-n<N>-<lod>-<mode>-c<client>-s<server>-<GPU>.csv`).
