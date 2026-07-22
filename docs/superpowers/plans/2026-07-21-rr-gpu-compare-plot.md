# plot_gpu_compare.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `research/scripts/plot_gpu_compare.py`, a standalone script that renders one self-contained figure per (metric × N-group) overlaying several GPUs' remote-rendering time series, with stable per-GPU identity and color across every figure.

**Architecture:** A single Python file consuming the fixed `rr-client --benchmark` CSV schema. GPU name and particle-count (N) are parsed from each CSV's filename. `--metric fps|latency` picks the drawer; one figure = one metric at one N-group with one curve per GPU, optional camera-phase shading, and an auto rule that puts a shared N in the title but a differing N (the max-N group) per legend entry. Run it 6× to produce the 2×3 grid; panels are assembled externally.

**Tech Stack:** Python 3.12, pandas, matplotlib, pytest. Run from a project-local venv (Ubuntu's system Python is externally managed).

## Global Constraints

- Standalone file — do **not** import from `research/scripts/plot_benchmark.py`; copy the phase-shading helpers so the script has no intra-repo dependency.
- Consume the existing `--benchmark` CSV schema **as-is**; no client/protocol/CSV changes.
- Metrics supported: `fps` and `latency` only.
- No summary table, twin-axis, or bitrate/encode/decode curves (that belongs to `plot_benchmark.py`).
- GPU→color map is fixed and colorblind-safe (Okabe-Ito): `A100 #0072B2`, `H200 #E69F00`, `RTX PRO 6000 #009E73`, `B300 #D55E00`. A given GPU is the same hue in all figures.
- Canonical GPU names: `A100`, `H200`, `RTX PRO 6000`, `B300`.
- CSV column order (header written by `rr-client.cpp`):
  `time_s,fps,kbps,server_ms,server_ms_std,compute_ms,render_ms,decode_ms,decode_ms_std,lat_mean_ms,lat_std_ms,lat_p50_ms,lat_p95_ms,lat_max_ms,lost,ctrl_events,phase`
- Filename grammar (from `benchmarkCsvPath`): `<prefix>-<YYYYMMDD>-rr-<role>-n<COUNT>-lod<N|off>-<light>-c<CLIENT>-s<SERVER>-<GPU>.csv`, where `<COUNT>` is a `countTag` like `100M`/`1G`/`2.1G` and `<GPU>` is a `gpuTag` (vendor/arch words already stripped, uppercased alnum + dashes, e.g. `A100-SXM4-40GB`, `RTXPRO6000`, `B300-SXM6`, `H200-SXM5-141GB`).

---

### Task 1: Environment + run-metadata parsing (GPU name, particle count)

**Files:**
- Create: `research/scripts/plot_gpu_compare.py`
- Create: `research/scripts/test_plot_gpu_compare.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `canonical_gpu(text: str) -> str` — collapse to uppercase alnum, return the first matching canonical name among `A100`, `H200`, `RTX PRO 6000` (key `RTXPRO6000`), `B300`; else the last `-`-delimited token of `text`.
  - `parse_count(tag: str) -> float` — `"100M"->1e8`, `"1G"->1e9`, `"2.1G"->2.1e9`, `"0"->0.0`, unparseable `->0.0`.
  - `parse_run_meta(path: str) -> dict` — `{"gpu": str, "n": float, "n_tag": str, "path": str}`; `n_tag` is the raw count tag (e.g. `"100M"`), `n` its float; `gpu` from `canonical_gpu(stem)`.

- [ ] **Step 1: Create the venv and install deps**

Run:
```bash
cd /home/cnavarro/temporal/mimir
python3 -m venv research/.venv
research/.venv/bin/pip install -q --upgrade pip
research/.venv/bin/pip install -q pandas matplotlib pytest
grep -qxF 'research/.venv/' .gitignore || echo 'research/.venv/' >> .gitignore
```
Expected: installs succeed; `.gitignore` contains `research/.venv/`.

- [ ] **Step 2: Write the failing test**

Create `research/scripts/test_plot_gpu_compare.py`:
```python
import math
import plot_gpu_compare as p


def test_canonical_gpu_known_models():
    assert p.canonical_gpu("A100-SXM4-40GB") == "A100"
    assert p.canonical_gpu("H200-SXM5-141GB") == "H200"
    assert p.canonical_gpu("RTXPRO6000") == "RTX PRO 6000"
    assert p.canonical_gpu("B300-SXM6") == "B300"


def test_canonical_gpu_unknown_falls_back_to_last_token():
    assert p.canonical_gpu("some-weird-V100") == "V100"


def test_parse_count():
    assert p.parse_count("100M") == 1e8
    assert p.parse_count("1G") == 1e9
    assert p.parse_count("2.1G") == 2.1e9
    assert p.parse_count("6G") == 6e9
    assert p.parse_count("0") == 0.0
    assert p.parse_count("garbage") == 0.0


def test_parse_run_meta_from_filename():
    name = "bench-20260721-rr-client-n100M-lod256-pt-cWS-sSRV-A100-SXM4-40GB.csv"
    meta = p.parse_run_meta("/tmp/" + name)
    assert meta["gpu"] == "A100"
    assert meta["n_tag"] == "100M"
    assert meta["n"] == 1e8
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plot_gpu_compare'` (or `AttributeError`).

- [ ] **Step 4: Write the minimal implementation**

Create `research/scripts/plot_gpu_compare.py` (module docstring + these definitions):
```python
#!/usr/bin/env python3
"""Overlay several GPUs' remote-rendering benchmark time series in one figure.

Consumes the CSV written by ``rr-client --benchmark F`` (see rr-client.cpp). One
invocation renders ONE self-contained figure: a chosen metric (fps or latency) at one
problem size, with one curve per GPU. GPU name and particle-count are parsed from each
CSV's filename. Run it 6 times (2 metrics x 3 N-groups) to build the full 2x3 grid;
panels are assembled externally so each stays independently shareable.

Usage:
    plot_gpu_compare.py --metric fps a100.csv h200.csv rtxpro.csv b300.csv -o fps_med.pdf
    plot_gpu_compare.py --metric latency *.csv --ylim 0 120 --no-show -o lat_max.pdf

Requires: pandas, matplotlib (pip install pandas matplotlib).
"""

import argparse
import math
import re
import sys
from pathlib import Path

# Canonical short name per GPU, matched as an alnum-collapsed substring of the filename.
# gpuTag() in remote_protocol.hpp already strips vendor/arch words, so these tokens survive.
GPU_KEYS = [
    ("A100", "A100"),
    ("H200", "H200"),
    ("RTXPRO6000", "RTX PRO 6000"),
    ("B300", "B300"),
]

_COUNT_SUFFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def canonical_gpu(text):
    """Map a filename (or GPU tag) to a canonical short name, else its last '-' token."""
    collapsed = "".join(ch for ch in text.upper() if ch.isalnum())
    for key, label in GPU_KEYS:
        if key in collapsed:
            return label
    return text.rsplit("-", 1)[-1]


def parse_count(tag):
    """countTag string ('100M', '2.1G', '0') -> particle count as a float; 0.0 if unparseable."""
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", tag.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _COUNT_SUFFIX[m.group(2)]


def parse_run_meta(path):
    """Extract {gpu, n, n_tag, path} from a --benchmark CSV filename."""
    stem = Path(path).stem
    m = re.search(r"-n([0-9]+(?:\.[0-9]+)?[KMGT]?)-lod", stem)
    n_tag = m.group(1) if m else ""
    return {
        "gpu": canonical_gpu(stem),
        "n": parse_count(n_tag) if n_tag else 0.0,
        "n_tag": n_tag,
        "path": path,
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add research/scripts/plot_gpu_compare.py research/scripts/test_plot_gpu_compare.py .gitignore
git commit -m "feat(rr-plot): run-metadata parsing for GPU-compare figures"
```

---

### Task 2: CSV loading + N-formatting + shared-N detection

**Files:**
- Modify: `research/scripts/plot_gpu_compare.py`
- Modify: `research/scripts/test_plot_gpu_compare.py`

**Interfaces:**
- Consumes: `parse_run_meta` (Task 1).
- Produces:
  - `load(path: str) -> pandas.DataFrame` — reads the CSV, adds `df["t"] = time_s - time_s[0]`, sets `df.attrs` to the `parse_run_meta` dict.
  - `fmt_n(n: float) -> str` — matplotlib mathtext: exact power of ten → `$10^{k}$`; else `$m.m\times10^{k}$`; `n<=0` → `"?"`.
  - `shared_n_tag(runs: list) -> str | None` — the common `n_tag` if all runs share it, else `None`.
  - `legend_label(df, shared) -> str` — `gpu` when `shared` is not None, else `f"{gpu} (N={fmt_n(n)})"`.

- [ ] **Step 1: Write the failing test**

Append to `research/scripts/test_plot_gpu_compare.py`:
```python
import pandas as pd  # noqa: E402

HEADER = ("time_s,fps,kbps,server_ms,server_ms_std,compute_ms,render_ms,decode_ms,"
          "decode_ms_std,lat_mean_ms,lat_std_ms,lat_p50_ms,lat_p95_ms,lat_max_ms,"
          "lost,ctrl_events,phase")


def _write_csv(tmp_path, gpu_tag, n_tag, rows):
    """rows: list of (t, fps, lat_mean, lat_std, phase). Fills unused columns with 0."""
    name = f"bench-20260721-rr-client-n{n_tag}-lod256-pt-cWS-sSRV-{gpu_tag}.csv"
    path = tmp_path / name
    lines = [HEADER]
    for t, fps, lat, std, phase in rows:
        lines.append(f"{t},{fps},0,0,0,0,0,0,0,{lat},{std},0,0,0,0,0,{phase}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_zeroes_time_and_tags(tmp_path):
    csv = _write_csv(tmp_path, "A100-SXM4-40GB", "100M",
                     [(10.0, 60, 20, 2, "far"), (11.0, 58, 22, 3, "orbit")])
    df = p.load(csv)
    assert df["t"].iloc[0] == 0.0
    assert df["t"].iloc[1] == 1.0
    assert df.attrs["gpu"] == "A100"
    assert df.attrs["n"] == 1e8


def test_fmt_n():
    assert p.fmt_n(1e8) == r"$10^{8}$"
    assert p.fmt_n(1e9) == r"$10^{9}$"
    s = p.fmt_n(2.1e9)
    assert "2.1" in s and "9" in s
    assert p.fmt_n(0) == "?"


def test_shared_n_tag_and_legend(tmp_path):
    a = p.load(_write_csv(tmp_path, "A100-SXM4-40GB", "1G", [(0.0, 60, 20, 2, "far")]))
    b = p.load(_write_csv(tmp_path, "B300-SXM6", "1G", [(0.0, 90, 12, 1, "far")]))
    assert p.shared_n_tag([a, b]) == "1G"
    assert p.legend_label(a, "1G") == "A100"

    b2 = p.load(_write_csv(tmp_path, "B300-SXM6", "6G", [(0.0, 90, 12, 1, "far")]))
    assert p.shared_n_tag([a, b2]) is None
    assert p.legend_label(b2, None) == f"B300 (N={p.fmt_n(6e9)})"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: FAIL — `AttributeError: module 'plot_gpu_compare' has no attribute 'load'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `research/scripts/plot_gpu_compare.py` (after `parse_run_meta`; add `import pandas as pd` to the imports block):
```python
def load(path):
    """Read one benchmark CSV, zero its time axis, and tag it with parsed run metadata."""
    df = pd.read_csv(path)
    df["t"] = df["time_s"] - df["time_s"].iloc[0]
    df.attrs.update(parse_run_meta(path))
    return df


def fmt_n(n):
    """Format a particle count as mathtext: 1e8 -> $10^{8}$, 2.1e9 -> $2.1\\times10^{9}$."""
    if n <= 0:
        return "?"
    exp = int(math.floor(math.log10(n)))
    mant = n / (10.0 ** exp)
    if abs(mant - 1.0) < 1e-9:
        return rf"$10^{{{exp}}}$"
    return rf"${mant:.1f}\times10^{{{exp}}}$"


def shared_n_tag(runs):
    """The common n_tag if every run shares it, else None (the max-N group differs per GPU)."""
    tags = {df.attrs["n_tag"] for df in runs}
    return runs[0].attrs["n_tag"] if len(tags) == 1 else None


def legend_label(df, shared):
    """GPU alone when N is shared (N goes in the title); GPU + its own N when N differs."""
    gpu = df.attrs["gpu"]
    return gpu if shared is not None else f"{gpu} (N={fmt_n(df.attrs['n'])})"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add research/scripts/plot_gpu_compare.py research/scripts/test_plot_gpu_compare.py
git commit -m "feat(rr-plot): CSV load, N formatting, shared-N legend rule"
```

---

### Task 3: Colors + phase-shading helpers

**Files:**
- Modify: `research/scripts/plot_gpu_compare.py`
- Modify: `research/scripts/test_plot_gpu_compare.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `GPU_COLORS: dict[str, str]` and `color_for(gpu: str) -> str` (fixed hue per canonical GPU; deterministic fallback for unknowns).
  - `PHASE_COLORS: dict[str, str]`, `PHASE_LABELS: dict[str, str]`.
  - `shade_phases(ax, df)` and `label_phases(ax, df)` — copied from `plot_benchmark.py` (background spans + in-band labels for the scripted camera phases). Guarded to no-op when `phase` is absent.

- [ ] **Step 1: Write the failing test**

Append to `research/scripts/test_plot_gpu_compare.py`:
```python
def test_color_for_stable_per_gpu():
    assert p.color_for("A100") == "#0072B2"
    assert p.color_for("B300") == "#D55E00"
    # Unknown GPUs are deterministic and distinct from each other.
    assert p.color_for("V100") == p.color_for("V100")
    assert p.color_for("V100") != p.color_for("MI300")


def test_phase_tables_present():
    assert "orbit" in p.PHASE_COLORS
    assert p.PHASE_LABELS["orbit"].startswith("Orbit")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: FAIL — `AttributeError: ... 'color_for'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `research/scripts/plot_gpu_compare.py` (color block near the top constants; phase block after `legend_label`):
```python
# Fixed hue per GPU (Okabe-Ito, colorblind-safe) so a GPU reads identically across all figures.
GPU_COLORS = {
    "A100":         "#0072B2",  # blue
    "H200":         "#E69F00",  # orange
    "RTX PRO 6000": "#009E73",  # green
    "B300":         "#D55E00",  # vermillion
}
# Deterministic fallback for any GPU not in the fixed map (also Okabe-Ito).
_FALLBACK_COLORS = ["#CC79A7", "#56B4E9", "#F0E442", "#999999", "#000000"]
_fallback_assigned = {}


def color_for(gpu):
    """Fixed color for a known GPU; a stable per-name fallback color otherwise."""
    if gpu in GPU_COLORS:
        return GPU_COLORS[gpu]
    idx = _fallback_assigned.setdefault(gpu, len(_fallback_assigned))
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


# Scripted camera phases: shaded behind the curves and named in-band. Legacy tokens map to
# their nearest current phase so older CSVs still plot. (Copied from plot_benchmark.py.)
PHASE_COLORS = {
    "far": "#d9d9d9", "orbit": "#c6dbef", "zoom_in": "#c7e9c0",
    "look_around": "#fdd0a2", "inside": "#dadaeb", "done": "#f0f0f0",
    "outside": "#d9d9d9", "idle": "#d9d9d9", "zoom_out": "#dadaeb",
}
PHASE_LABELS = {
    "far": "Far view (Static)", "orbit": "Orbit view (Mid Dynamic)",
    "zoom_in": "Zoom in (Low Dynamic)", "look_around": "Look Around (High Dynamic)",
    "inside": "Inside view (Static)",
    "outside": "Far view (Static)", "idle": "Far view (Static)", "zoom_out": "Zoom out",
}


def shade_phases(ax, df):
    """Shade contiguous same-phase spans behind the curves (no-op without a phase column)."""
    if "phase" not in df.columns:
        return
    start = 0
    phase = df["phase"].iloc[0]
    for i in range(1, len(df) + 1):
        cur = df["phase"].iloc[i] if i < len(df) else None
        if cur != phase:
            color = PHASE_COLORS.get(str(phase), "#eeeeee")
            ax.axvspan(df["t"].iloc[start], df["t"].iloc[i - 1], color=color,
                       alpha=0.5, lw=0, zorder=0)
            start, phase = i, cur


def label_phases(ax, df):
    """Write each phase's display name inside its band as muted background text."""
    if "phase" not in df.columns:
        return
    trans = ax.get_xaxis_transform()
    start = 0
    phase = df["phase"].iloc[0]
    for i in range(1, len(df) + 1):
        cur = df["phase"].iloc[i] if i < len(df) else None
        if cur != phase:
            if str(phase) != "done":
                xmid = 0.5 * (df["t"].iloc[start] + df["t"].iloc[i - 1])
                text = PHASE_LABELS.get(str(phase), str(phase)).replace(" (", "\n(", 1)
                ax.text(xmid, 0.97, text, transform=trans, ha="center", va="top",
                        fontsize=7.5, style="italic", color="#666666", alpha=0.7,
                        linespacing=1.4, zorder=1)
            start, phase = i, cur
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add research/scripts/plot_gpu_compare.py research/scripts/test_plot_gpu_compare.py
git commit -m "feat(rr-plot): fixed GPU palette + phase-shading helpers"
```

---

### Task 4: Metric drawers (fps line; latency mean + std band)

**Files:**
- Modify: `research/scripts/plot_gpu_compare.py`
- Modify: `research/scripts/test_plot_gpu_compare.py`

**Interfaces:**
- Consumes: `color_for`, `legend_label` (Tasks 2–3).
- Produces:
  - `draw_fps(ax, runs, shared)` — one solid `fps` line per run, colored by GPU, labeled via `legend_label`; sets the y-label.
  - `draw_latency(ax, runs, shared)` — `lat_mean_ms` line per run plus a translucent `±lat_std_ms` band (alpha 0.12, `zorder=0.5`); sets the y-label.
  - `METRICS: dict[str, tuple]` mapping `"fps"`/`"latency"` to `(drawer, nice_name)`.

The `matplotlib` import at module top must select a headless-safe backend before `pyplot`:
```python
import matplotlib
import os
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")   # headless / CI: render to file without a display
import matplotlib.pyplot as plt
```

- [ ] **Step 1: Write the failing test**

Append to `research/scripts/test_plot_gpu_compare.py`:
```python
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_draw_fps_one_line_per_run(tmp_path):
    a = p.load(_write_csv(tmp_path, "A100-SXM4-40GB", "1G",
                          [(0.0, 60, 20, 2, "far"), (1.0, 61, 21, 2, "orbit")]))
    b = p.load(_write_csv(tmp_path, "B300-SXM6", "1G",
                          [(0.0, 120, 8, 1, "far"), (1.0, 118, 9, 1, "orbit")]))
    fig, ax = plt.subplots()
    p.draw_fps(ax, [a, b], "1G")
    assert len(ax.get_lines()) == 2
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert set(labels) == {"A100", "B300"}
    plt.close(fig)


def test_draw_latency_adds_band(tmp_path):
    a = p.load(_write_csv(tmp_path, "A100-SXM4-40GB", "1G",
                          [(0.0, 60, 20, 2, "far"), (1.0, 61, 22, 3, "orbit")]))
    fig, ax = plt.subplots()
    p.draw_latency(ax, [a], "1G")
    assert len(ax.get_lines()) == 1               # the mean line
    assert len(ax.collections) >= 1               # the fill_between std band
    plt.close(fig)


def test_metrics_table():
    assert set(p.METRICS) == {"fps", "latency"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: FAIL — `AttributeError: ... 'draw_fps'`.

- [ ] **Step 3: Write the minimal implementation**

Add the headless-backend lines to the import block (as shown above), then add after the phase helpers:
```python
def draw_fps(ax, runs, shared):
    """One solid FPS line per GPU."""
    for df in runs:
        ax.plot(df["t"], df["fps"], color=color_for(df.attrs["gpu"]), lw=1.7,
                label=legend_label(df, shared))
    ax.set_ylabel("frames per second (FPS)")


def draw_latency(ax, runs, shared):
    """Mean end-to-end latency per GPU, with a translucent +/-1 std-dev band behind it."""
    for df in runs:
        c = color_for(df.attrs["gpu"])
        ax.plot(df["t"], df["lat_mean_ms"], color=c, lw=1.7,
                label=legend_label(df, shared))
        if "lat_std_ms" in df.columns:
            lo = (df["lat_mean_ms"] - df["lat_std_ms"]).clip(lower=0)
            hi = df["lat_mean_ms"] + df["lat_std_ms"]
            ax.fill_between(df["t"], lo, hi, color=c, alpha=0.12, lw=0, zorder=0.5)
    ax.set_ylabel("end-to-end latency (ms)")


METRICS = {
    "fps":     (draw_fps, "FPS"),
    "latency": (draw_latency, "Latency"),
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add research/scripts/plot_gpu_compare.py research/scripts/test_plot_gpu_compare.py
git commit -m "feat(rr-plot): fps and latency metric drawers"
```

---

### Task 5: Figure assembly + CLI (end-to-end)

**Files:**
- Modify: `research/scripts/plot_gpu_compare.py`
- Modify: `research/scripts/test_plot_gpu_compare.py`

**Interfaces:**
- Consumes: `load`, `shared_n_tag`, `fmt_n`, `METRICS`, `shade_phases`, `label_phases` (Tasks 2–4).
- Produces:
  - `plot(runs, metric, out=None, show=True, title=None, ylim=None, logy=False, phases=True) -> matplotlib.figure.Figure`.
  - `main(argv=None) -> int` — argparse CLI; returns 0 on success.

- [ ] **Step 1: Write the failing test**

Append to `research/scripts/test_plot_gpu_compare.py`:
```python
def _four_gpu_csvs(tmp_path, n_tag="1G"):
    rows = [(0.0, 60, 20, 2, "far"), (1.0, 61, 22, 3, "orbit"), (2.0, 59, 21, 2, "inside")]
    tags = ["A100-SXM4-40GB", "H200-SXM5-141GB", "RTXPRO6000", "B300-SXM6"]
    return [_write_csv(tmp_path, t, n_tag, rows) for t in tags]


def test_plot_title_uses_shared_n(tmp_path):
    runs = [p.load(c) for c in _four_gpu_csvs(tmp_path, "100M")]
    fig = p.plot(runs, "fps", show=False, phases=True)
    assert p.fmt_n(1e8) in fig.axes[0].get_title()
    plt.close(fig)


def test_main_writes_file_both_metrics(tmp_path):
    csvs = _four_gpu_csvs(tmp_path, "1G")
    for metric in ("fps", "latency"):
        out = tmp_path / f"{metric}.pdf"
        rc = p.main(["--metric", metric, *csvs, "-o", str(out), "--no-show"])
        assert rc == 0
        assert out.exists() and out.stat().st_size > 0


def test_main_labels_count_mismatch_errors(tmp_path):
    csvs = _four_gpu_csvs(tmp_path, "1G")
    import pytest
    with pytest.raises(SystemExit):
        p.main(["--metric", "fps", *csvs, "--labels", "only,two", "--no-show"])


def test_main_differing_n_puts_n_in_legend(tmp_path):
    a = _write_csv(tmp_path, "A100-SXM4-40GB", "1G", [(0.0, 60, 20, 2, "far")])
    b = _write_csv(tmp_path, "B300-SXM6", "6G", [(0.0, 120, 8, 1, "far")])
    out = tmp_path / "maxn.pdf"
    rc = p.main(["--metric", "fps", a, b, "-o", str(out), "--no-show"])
    assert rc == 0 and out.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: FAIL — `AttributeError: ... 'plot'` / `'main'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `research/scripts/plot_gpu_compare.py` (after `METRICS`):
```python
def plot(runs, metric, out=None, show=True, title=None, ylim=None, logy=False, phases=True):
    """Render one metric for all runs into a single self-contained figure."""
    draw, nice = METRICS[metric]
    shared = shared_n_tag(runs)
    fig, ax = plt.subplots(figsize=(8, 5))
    if phases:
        shade_phases(ax, runs[0])       # background, before the curves
    draw(ax, runs, shared)
    if phases:
        label_phases(ax, runs[0])       # text, on top of the shading
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    elif not logy:
        ax.set_ylim(bottom=0)
    if title is None:
        title = nice if shared is None else f"{nice} — N = {fmt_n(runs[0].attrs['n'])}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    if show:
        plt.show()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", required=True, choices=["fps", "latency"],
                    help="which metric this figure plots")
    ap.add_argument("csv", nargs="+", help="benchmark CSV(s) for ONE N-group (one per GPU)")
    ap.add_argument("-o", "--out", help="save the figure to this path (e.g. fps_med.pdf)")
    ap.add_argument("--labels", help="comma-separated GPU labels overriding the parsed names, "
                                     "in CSV order")
    ap.add_argument("--ylim", nargs=2, type=float, metavar=("LO", "HI"),
                    help="pin the y-axis (share a scale across a metric row)")
    ap.add_argument("--logy", action="store_true", help="log-scale the y-axis")
    ap.add_argument("--no-phases", dest="phases", action="store_false",
                    help="disable camera-phase shading")
    ap.add_argument("-t", "--title", help="override the figure title")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="don't open a plot window")
    args = ap.parse_args(argv)

    for c in args.csv:
        if not Path(c).exists():
            ap.error(f"no such file: {c}")
    runs = [load(c) for c in args.csv]

    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(runs):
            ap.error(f"--labels has {len(labels)} entries but {len(runs)} CSVs given")
        for df, lbl in zip(runs, labels):
            df.attrs["gpu"] = lbl

    plot(runs, args.metric, out=args.out, show=args.show, title=args.title,
         ylim=tuple(args.ylim) if args.ylim else None, logy=args.logy, phases=args.phases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd research/scripts && ../.venv/bin/python -m pytest test_plot_gpu_compare.py -v`
Expected: PASS (full suite).

- [ ] **Step 5: Manual smoke check (both metrics render to a file)**

Run:
```bash
cd /home/cnavarro/temporal/mimir/research/scripts
../.venv/bin/python - <<'PY'
import tempfile, os
from test_plot_gpu_compare import _write_csv
import plot_gpu_compare as p
d = tempfile.mkdtemp()
rows = [(0.0,60,20,2,"far"),(1.0,61,22,3,"orbit"),(2.0,59,21,2,"inside")]
csvs = [_write_csv(__import__("pathlib").Path(d), t, "1G", rows)
        for t in ["A100-SXM4-40GB","H200-SXM5-141GB","RTXPRO6000","B300-SXM6"]]
for m in ("fps","latency"):
    p.main(["--metric",m,*csvs,"-o",os.path.join(d,f"{m}.pdf"),"--no-show"])
    print(m, os.path.getsize(os.path.join(d,f"{m}.pdf")), "bytes")
PY
```
Expected: prints two non-zero byte sizes; `wrote …/fps.pdf` and `wrote …/latency.pdf`.

- [ ] **Step 6: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add research/scripts/plot_gpu_compare.py research/scripts/test_plot_gpu_compare.py
git commit -m "feat(rr-plot): figure assembly + CLI for cross-GPU comparison plots"
```

---

## Post-implementation: finalize against a real CSV

Once a real `rr-client --benchmark` CSV from one of the four GPUs is available:

1. Confirm `parse_run_meta` returns the expected `gpu`/`n` for the real filename (the `<gpu>` token format is the one uncertainty). If a real GPU tag doesn't match a `GPU_KEYS` substring, adjust the key (not the canonical label).
2. Run one real figure per metric and eyeball legend, colors, phase bands, and the N-in-title vs N-in-legend behavior.
3. When invoking the `dataviz` skill for final palette polish, keep the Okabe-Ito hues unless a conflict is found; they are already colorblind-safe.
