#!/usr/bin/env python3
"""Batch-generate one benchmark GRID figure per rendering mode, reusing plot_benchmark.py.

This is the "mother" driver over ``plot_benchmark.py``: it discovers the per-run CSVs in a
data directory, groups them, and composes -- for each of the three rendering modes
(``pt`` = path tracing, ``phong`` = the impostor path, ``raster`` = unshaded /
--render-path flat) -- a single
2x3 grid figure:

                100 M            1 B (LOD 128)        Max N per GPU
            +----------------+----------------+----------------------+
  Throughput|  GPUs overlaid |  GPUs overlaid |  GPUs overlaid       |
            +----------------+----------------+----------------------+
  Latency   |  GPUs overlaid |  GPUs overlaid |  GPUs overlaid       |
            +----------------+----------------+----------------------+

Columns are particle-count regimes; each cell overlays every GPU that has a run for that
(mode, regime). The MAX column is special: each GPU ran the largest N it could fit
(A100 5G, RTX PRO 6000 6G, H200 9G, B300 16G), so the per-cell legend spells out each
GPU's N -- the other columns share one N, named in the column title instead.

Panels are drawn by plot_benchmark.py's own draw_throughput/draw_timings, so this script
never duplicates plotting logic -- it only decides *which* CSVs land in *which* cell and
lays the cells out. Add a run mode, a GPU, or a size and the grid picks it up from the
filenames; nothing here is hard-coded to today's exact runs beyond the column definitions.

CSV filename convention (written by rr-client --benchmark), fields joined by '-':
    mimir-<date>-rr-client-n<N>-<lod>-<mode>-c<client>-s<server>-<GPU...>.csv
e.g. mimir-20260722-rr-client-n1G-lod128-pt-cSGBOOK3-sNODEGPU02-RTXPRO6000.csv

Usage:
    plot_grid.py                                  # all 3 modes -> research/plots/*.pdf
    plot_grid.py --data-dir research/data -o research/plots
    plot_grid.py --modes pt phong                 # only these modes
    plot_grid.py --format png                     # PNG instead of PDF
    plot_grid.py --logy                           # log-scale the timings rows
    plot_grid.py --show                            # also open windows

Requires: pandas, matplotlib (same as plot_benchmark.py, which it imports).
"""

import argparse
import sys
from pathlib import Path

# Import the single-run plotter as a library so its panels stay the source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark as pb  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


# --- Rendering modes: filename token -> display name --------------------------------------
MODES = {
    "pt":     "Path tracing",
    "phong":  "Phong",
    "raster": "Unshaded (render-path flat)",
}

# Short form for the subtitle's "Rendered with ..." line -- MODES' names are the column-title
# style (lowercase, parenthetical detail); this is the title-case short form for prose.
RENDER_LABEL = {
    "pt":     "Path Tracing",
    "phong":  "Phong",
    "raster": "Unshaded",
}

# --- GPUs: a fixed order + short name (table/console) + legend fields + stable color, so a
# GPU reads the same in every cell of every grid. `key` is matched as a substring of the raw
# GPU field from the filename (e.g. "A100-SXM4-80GB" -> A100), so minor host-string
# differences don't matter. The legend's parenthetical carries client<->server location
# (client is always CL; see research/data/server-countries for the server side) -- the caption
# spells out what the country codes mean.
# (key, short name, legend device name, memory, server country, color)
GPUS = [
    ("A100",       "A100",         "A100 SXM4",        "80GB",  "RO", "#1f77b4"),
    ("H200",       "H200",         "H200 SXM5",         "141GB", "FR", "#ff7f0e"),
    ("RTXPRO6000", "RTX PRO 6000", "RTX PRO 6000 BSE", "96GB",  "CL", "#2ca02c"),
    ("B300",       "B300",         "B300 SXM6",         "288GB", "NL", "#d62728"),
]
GPU_ORDER = {g[0]: i for i, g in enumerate(GPUS)}
GPU_NAME  = {g[0]: g[1] for g in GPUS}
GPU_COLOR = {g[0]: g[5] for g in GPUS}

# Legend text is set in a monospace font (see cell_runs/draw_* legend calls) specifically so
# this padding lines up into columns: device name, then memory and CL<->country right-aligned
# in their own fields, so every GPU's "GB" and country code sit under each other.
_NAME_W = max(len(g[2]) for g in GPUS)
_MEM_W = max(len(g[3]) for g in GPUS)
GPU_LEGEND = {g[0]: f"{g[2]:<{_NAME_W}} ({g[3]:>{_MEM_W}}, CL <-> {g[4]})" for g in GPUS}

# --- Columns: (key, {size tokens it accepts}, column title). Size tokens are the CSV's
# n<size> field with the leading 'n' stripped. The MAX column collects each GPU's largest run.
COLUMNS = [
    ("100M", {"100M"},                "n=100M"),
    ("1G",   {"1G"},                  "n=1B (LOD 128)"),
    ("MAX",  {"5G", "6G", "9G", "16G"}, "n=Max (LOD 256)"),
]


def format_n(token):
    """CSV size token (e.g. '100M', '1G', '16G') -> paper-style label ('100 M', '1 B', '16 B')."""
    if token.endswith("G"):
        return f"{token[:-1]} B"
    if token.endswith("M"):
        return f"{token[:-1]} M"
    return token


def parse_meta(path):
    """Extract (mode, size, gpu_key) from a benchmark CSV filename, or None if it doesn't
    match the convention / a known GPU. size is the n<...> field without its leading 'n'."""
    tokens = Path(path).stem.split("-")
    mode = next((t for t in tokens if t in MODES), None)
    if mode is None:
        return None
    mi = tokens.index(mode)
    # size is the first n<digit...> token (e.g. n100M, n1G, n16G).
    size = next((t[1:] for t in tokens
                 if len(t) > 1 and t[0] == "n" and t[1].isdigit()), None)
    # GPU is everything after mode + client(c..) + server(s..): tokens[mi+3:], joined back.
    raw_gpu = "-".join(tokens[mi + 3:])
    gpu = next((k for k in GPU_ORDER if k in raw_gpu), None)
    if size is None or gpu is None:
        return None
    return {"mode": mode, "size": size, "gpu": gpu, "raw_gpu": raw_gpu}


def collect(data_dir):
    """Parse every CSV under data_dir into (meta, path) pairs, skipping non-matching files."""
    items = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        meta = parse_meta(p)
        if meta is not None:
            items.append((meta, p))
    return items


def cell_runs(items, mode, size_set, col_key):
    """The runs for one grid cell: every CSV matching this mode whose size is in size_set,
    ordered by GPU. Returns (runs, run_color) ready for pb.draw_*; labels are the full GPU
    legend name, plus the particle count in the MAX column where each GPU used a different N
    (the other columns share one N, already named in the column title)."""
    chosen = sorted(
        [(m, p) for (m, p) in items if m["mode"] == mode and m["size"] in size_set],
        key=lambda mp: GPU_ORDER[mp[0]["gpu"]],
    )
    runs, run_color = [], {}
    for meta, path in chosen:
        df = pb.load(path)
        name = GPU_LEGEND[meta["gpu"]]
        # Billion-particle count (5B/6B/9B/16B), compact (no space) so it doesn't read as
        # another "GB"-style byte unit next to the memory size earlier on the same line.
        # Right-justified so "n= 5B"/"n=16B" line up too.
        billions = f"{meta['size'][:-1]}B"
        label = f"{name} · n={billions:>3}" if col_key == "MAX" else name
        df.attrs["label"] = label
        run_color[label] = GPU_COLOR[meta["gpu"]]
        runs.append(df)
    return runs, run_color


def build_grid(items, mode, logy):
    """Compose the 2x3 grid figure for one rendering mode. Returns the figure, or None if
    this mode has no data at all."""
    if not any(m["mode"] == mode for m, _ in items):
        return None
    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(9.2 * len(COLUMNS), 12.5))
    for col_i, (col_key, size_set, col_title) in enumerate(COLUMNS):
        ax_tp, ax_lat = axes[0][col_i], axes[1][col_i]
        runs, run_color = cell_runs(items, mode, size_set, col_key)
        if not runs:
            for ax in (ax_tp, ax_lat):
                ax.set_axis_off()
            ax_tp.set_title(f"{col_title}\n(no runs)", fontsize=27, pad=18)
            continue
        pb.draw_throughput(ax_tp, runs, run_color, single=False, yrange_a=None)
        pb.draw_timings(ax_lat, runs, run_color, single=False, yrange_b=None, logy=logy)
        # The draw_* helpers set their own per-panel titles; replace them with the column
        # regime on the top row and drop the redundant one on the bottom row (the ms y-axis
        # already identifies it). The metric identity of each row is carried by its y-axis.
        ax_tp.set_title(col_title, fontsize=27, pad=18)
        ax_lat.set_title("")
        # The row already carries its unit via the leftmost column's y-axis label (Throughput /
        # Latency), and every column shares the same time axis named once under the bottom row
        # -- repeating "FPS"/"Latency (ms)"/"time (s)" in every cell just eats width for nothing.
        if col_i > 0:
            ax_tp.set_ylabel("")
            ax_lat.set_ylabel("")
        ax_tp.set_xlabel("")
    fig.suptitle("Mìmir Remote Rendering Benchmark", fontsize=31, y=0.995)
    subtitle = (f"Random moving Particles in 3D kmodal distribution, k=64, stdev=0.07 "
                f"-- Rendered with {RENDER_LABEL[mode]}")
    fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=23,
             style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    repo = here.parent.parent  # research/scripts -> research -> repo
    p.add_argument("--data-dir", default=str(repo / "research" / "data"),
                   help="directory of benchmark CSVs (default: research/data)")
    p.add_argument("-o", "--out-dir", default=str(repo / "research" / "plots"),
                   help="directory to write the grid figures into (default: research/plots)")
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES),
                   help="which rendering modes to plot (default: all three)")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                   help="output file format (default: pdf)")
    p.add_argument("--logy", action="store_true", help="log-scale the timings (ms) rows")
    p.add_argument("--show", action="store_true", help="also open a window per figure")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: no such data dir: {data_dir}", file=sys.stderr)
        return 1
    items = collect(data_dir)
    if not items:
        print(f"error: no benchmark CSVs matched the naming convention in {data_dir}",
              file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"found {len(items)} CSV(s) in {data_dir}")
    wrote = 0
    for mode in args.modes:
        fig = build_grid(items, mode, args.logy)
        if fig is None:
            print(f"  {mode:<7} no data -- skipped")
            continue
        # Per-cell census, so it's obvious what landed where (and what's missing).
        for col_key, size_set, _ in COLUMNS:
            gpus = sorted({GPU_NAME[m["gpu"]] for m, _ in items
                           if m["mode"] == mode and m["size"] in size_set},
                          key=lambda n: [GPU_NAME[k] for k in GPU_ORDER].index(n))
            print(f"  {mode:<7} {col_key:<4}: {', '.join(gpus) if gpus else '(none)'}")
        out = out_dir / f"benchmark-grid-{mode}.{args.format}"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  -> wrote {out}")
        wrote += 1
    if args.show and wrote:
        plt.show()
    print(f"done: {wrote} grid figure(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
