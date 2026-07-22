# Remote-rendering GPU-comparison plotting script — design

**Date:** 2026-07-21
**Status:** Approved (brainstorming)
**Component:** `research/scripts/plot_gpu_compare.py` (new)

## Purpose

Generate one self-contained figure per (metric × N-group) for a cross-GPU remote-rendering
benchmark. Each figure overlays several GPUs' time series for a single metric at a single
problem size, with GPU identity and color kept stable across every figure so the same GPU
reads identically in all of them.

The benchmark spans **4 GPUs** — A100 40GB SXM, H200 141GB SXM, RTX PRO 6000 Blackwell SE
96GB, B300 SXM — at **3 N-groups**:

- **medium** — N = 10^8 particles (common N across all GPUs)
- **large** — N = 10^9 particles (common N; A100 40GB is the memory floor, may OOM — an
  expected, reportable result rather than a bug)
- **max** — N = each GPU's own memory ceiling (N differs per GPU)

Two metrics (FPS, latency) × 3 N-groups = **6 figures**, each produced by one invocation.
Figures are assembled into a 2×3 grid (metric rows × N columns) externally (e.g. LaTeX), so
every panel stays independently shareable/screenshottable with its own legend, axes, title.

This is a *separate* script from the existing `research/scripts/plot_benchmark.py`. That
script is purpose-built for single-run / file-overlaid analysis (twin-axis throughput +
timings panels, color-by-filename, a summary table) and would be tangled by bolting on a
single-metric, color-by-GPU comparison mode. The new script is standalone and focused.

## Inputs

Consumes the CSV written by `rr-client --benchmark F` (see `rr-client.cpp`; schema is fixed).
Relevant columns:

```
time_s, fps, kbps, server_ms, server_ms_std, compute_ms, render_ms,
decode_ms, decode_ms_std, lat_mean_ms, lat_std_ms, lat_p50_ms, lat_p95_ms,
lat_max_ms, lost, ctrl_events, phase
```

Filenames self-describe the run:
`<prefix>-<date>-rr-client-c<client>-s<server>-<gpu>.csv`, and additionally encode the
particle_count and lod parameters (from `benchmarkCsvPath` in the client). The script parses
the **GPU name** and **particle_count (N)** from the filename.

One invocation takes the CSVs for **one N-group** (typically the 4 GPUs). Each CSV becomes
one curve.

## CLI

```
plot_gpu_compare.py --metric {fps|latency} CSV... \
    [-o OUT] [--labels A,B,...] [--ylim LO HI] [--logy] \
    [--no-phases] [-t TITLE] [--no-show]
```

- `--metric {fps|latency}` — which of the two metric figures to render. **Required.**
- `CSV...` — the input CSVs for a single N-group (one per GPU).
- `-o OUT` — save the figure (e.g. `.pdf`/`.png`); otherwise show a window.
- `--labels A,B,...` — override the auto-parsed per-CSV GPU labels (comma-separated, in CSV
  order).
- `--ylim LO HI` — pin the y-axis so all three N-plots in a metric row share one scale
  (makes the medium→large→max collapse visually honest). Auto-scaled if omitted.
- `--logy` — log-scale the y-axis (for wide dynamic range across N).
- `--no-phases` — disable camera-phase shading.
- `-t TITLE` — override the auto title.
- `--no-show` — don't open a window (table/figure-file only; Agg-safe for headless runs).

Run it 6 times (2 metrics × 3 N-groups) to produce the full grid.

## Behavior / rendering

**Curve identity = GPU.** Parsed from the filename `-s<server>-<gpu>` field, normalized to
short canonical names: `A100`, `H200`, `RTX PRO 6000`, `B300`. `--labels` overrides. The
normalization map tolerates vendor-prefixed strings (e.g. `"NVIDIA B300"`); its exact keys
are finalized against a real CSV's `<gpu>` token before implementation is considered done.

**Stable color per GPU.** A fixed `GPU -> color` dict, so a given GPU is the same hue in all
6 figures. Palette is colorblind-safe (apply the `dataviz` skill's categorical palette when
implementing). A GPU not in the dict falls back to a cycle color (deterministic by sorted
name) and emits a warning.

**Metric: `fps`.** One solid line per GPU (`fps` column). x = time (s), zeroed at start.

**Metric: `latency`.** Per GPU: a `lat_mean_ms` solid line plus a translucent `±lat_std_ms`
band (alpha ~0.12, drawn *behind* the mean lines) to show jitter/stability. Tail percentiles
(p95/max) are not plotted here.

**Phase shading (default on).** All GPUs run the same scripted camera path, so phases align
across CSVs. Shade contiguous same-phase spans once behind the curves and label each band
in-place, reusing the `PHASE_COLORS` / `PHASE_LABELS` semantics and the `shade_phases` /
`label_phases` logic from `plot_benchmark.py` (**copied** into this script, not imported, to
keep it standalone). The phase column is taken from the first CSV; a warning is emitted if
the input CSVs disagree on their phase sequence. `--no-phases` disables it.

**N placement (one auto rule).**
- All input CSVs share the same particle_count → N goes in the **title**
  (e.g. `FPS — N = 10^8`).
- particle_counts differ (the max-N group) → N is appended **per legend entry**
  (e.g. `B300 (N = 2.1×10^9)`).

**Axes.** x = time (s). y auto-scaled unless `--ylim`/`--logy` given. Grid at low alpha.
Legend shows GPU (+ N suffix when applicable).

## Structure

Single self-contained file mirroring `plot_benchmark.py`'s idioms:

- `load(path)` — read one CSV, zero `time_s` to `t`, parse GPU + N from the filename, stash
  on `df.attrs`.
- `canonical_gpu(raw)` — normalize a raw `<gpu>` token to a canonical short name.
- `GPU_COLORS` — the fixed name→color dict.
- `PHASE_COLORS` / `PHASE_LABELS` / `shade_phases` / `label_phases` — copied from
  `plot_benchmark.py`.
- `draw_fps(ax, runs, ...)` and `draw_latency(ax, runs, ...)` — the two per-metric drawers.
- `plot(runs, metric, ...)` — build the single-axes figure, title/legend, save/show.
- `main()` — argparse, load inputs, dispatch by `--metric`.

No summary table (figures only; `plot_benchmark.py` owns tabular output).

## Testing

A pytest-style test using tiny synthetic CSV fixtures (2–3 fabricated GPU CSVs with known N,
a couple of phases, a few time rows), written to a tmp dir with realistic filenames. Cover:

- GPU name + N parsed correctly from filenames; `canonical_gpu` maps known vendor strings.
- N-in-title vs N-in-legend switch fires correctly for same-N vs differing-N input sets.
- Both `--metric fps` and `--metric latency` render without error and honor `-o` (file
  written, non-empty).
- `--labels` override and `--no-phases` take effect.
- Runs headless (Agg backend / `--no-show`), no display required.

## Out of scope (YAGNI)

- No automatic 2×3 grid assembly — panels are composed externally so each stays shareable.
- No new CSV columns or client changes — consumes the existing `--benchmark` schema as-is.
- No summary tables, twin-axis, or bitrate/encode/decode curves (that's `plot_benchmark.py`).
- No support for metrics beyond fps/latency in this script.
