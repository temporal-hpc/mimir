#!/usr/bin/env python3
"""Plot the remote-rendering benchmark time series and print a summary table.

Consumes the CSV written by ``rr-client --benchmark F`` (see rr-client.cpp). Columns:

    time_s, fps, kbps, server_ms, decode_ms,
    lat_mean_ms, lat_p50_ms, lat_p95_ms, lat_max_ms, lost, ctrl_events, phase

Only the metrics worth watching *over time* get a curve; the rest live in a table of
per-phase (and overall) averages. Two time-series panels, both grouping curves that share a
Y-axis magnitude (or split onto twin axes when they do not):

    1. Throughput   fps (left axis)  +  bitrate kbps (right axis)              -- twin Y
    2. Timings (ms) mean end-to-end latency
                    + server encode/readback + client decode                  -- shared ms

Everything else (latency p50/p95/max, frame loss, control events) is reported in the table
only. `server_ms` is the server's per-frame production cost: NVENC *encode* time for an
H.264 stream, or framebuffer *readback* time for a raw stream.

Usage:
    plot_benchmark.py run.csv                       # one run: panels are phase-shaded
    plot_benchmark.py a.csv b.csv c.csv             # compare runs: overlaid, legend by file
    plot_benchmark.py run.csv -o run.png            # save instead of (also) showing
    plot_benchmark.py run.csv -t "My title"         # override the figure title
    plot_benchmark.py run.csv --no-show             # table only, no window

Requires: pandas, matplotlib (pip install pandas matplotlib).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Metrics that get a time-series curve are defined by the panels below; everything else in
# the CSV is summarised in the table only. Phase spans are shaded with these colors and named
# in-band (muted, as background info) under their display names. The legacy tokens (idle/zoom_out
# from older CSVs) share styling with their current equivalents so old runs still plot.
PHASE_COLORS = {
    "far":         "#d9d9d9",
    "orbit":       "#c6dbef",
    "zoom_in":     "#c7e9c0",
    "look_around": "#fdd0a2",
    "inside":      "#dadaeb",
    "done":        "#f0f0f0",
    # Legacy tokens from earlier CSVs, styled like their nearest current phase.
    "outside":     "#d9d9d9",
    "idle":        "#d9d9d9",
    "zoom_out":    "#dadaeb",
}
PHASE_LABELS = {
    "far":         "Fixed Far (Static)",
    "orbit":       "Orbit (Mid Dynamic)",
    "zoom_in":     "Zoom in (Low Dynamic)",
    "look_around": "Look Around (High Dynamic)",
    "inside":      "Fixed Inside (Static)",
    # Legacy tokens from earlier CSVs.
    "outside":     "Fixed Far (Static)",
    "idle":        "Fixed Far (Static)",
    "zoom_out":    "Zoom out",
}


def load(path):
    """Read one benchmark CSV, normalise time to start at 0, tag it with a short label."""
    df = pd.read_csv(path)
    df["t"] = df["time_s"] - df["time_s"].iloc[0]
    df.attrs["label"] = Path(path).stem
    return df


def shade_phases(ax, df):
    """Shade contiguous same-phase spans behind a single run's curves.

    The "Client Camera" legend key is built separately (from the phases present), so shading
    here stays unlabelled and never pollutes a panel's own curve legend.
    """
    if "phase" not in df.columns:
        return
    start = 0
    phase = df["phase"].iloc[0]
    for i in range(1, len(df) + 1):
        cur = df["phase"].iloc[i] if i < len(df) else None
        if cur != phase:
            color = PHASE_COLORS.get(str(phase), "#eeeeee")
            t0 = df["t"].iloc[start]
            t1 = df["t"].iloc[i - 1]
            ax.axvspan(t0, t1, color=color, alpha=0.5, lw=0, zorder=0)
            start, phase = i, cur


def label_phases(ax, df):
    """Write each phase's display name horizontally inside its shaded band, as muted background
    info — replaces a separate phase legend, so the name sits right on the color it describes.
    The "(motion class)" tag drops to a second line to stay narrow enough for a 12 s band.
    """
    if "phase" not in df.columns:
        return
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
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


# Timings panel (all ms, shared axis): mean end-to-end latency plus the two per-frame pipeline
# costs. (csv column, legend name, linewidth, linestyle, single-run color). Latency percentiles
# (p50/p95) and max stay in the averages table rather than crowding the plot.
TIMING_SERIES = [
    ("lat_mean_ms", "latency mean", 1.8, "-",              "#3182bd"),
    ("server_ms",   "encode",       1.5, (0, (3, 1, 1, 1)), "#e6550d"),
    ("decode_ms",   "decode",       1.5, (0, (1, 1)),       "#31a354"),
]


def plot(runs, out, show, title=None):
    single = len(runs) == 1
    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    # A stable color per run so the same file reads the same across both panels.
    run_color = {r.attrs["label"]: c for r, c in
                 zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"])}

    # --- Panel 1: throughput -- fps and bitrate live on separate magnitudes, so twin axes.
    ax_kbps = ax_tp.twinx()
    for df in runs:
        c = run_color[df.attrs["label"]]
        lbl = df.attrs["label"]
        ax_tp.plot(df["t"], df["fps"], color=c, lw=1.6,
                   label=f"{lbl} FPS" if not single else "Frames per second (FPS)")
        ax_kbps.plot(df["t"], df["kbps"], color=c, lw=1.1, ls="--",
                     label=f"{lbl} bitrate" if not single else "Bitrate (kbps)")
    ax_tp.set_ylabel("frames / s")
    ax_kbps.set_ylabel("bitrate (kbps, dashed)")
    ax_tp.set_title("Throughput: FPS (solid) + bitrate (dashed)")
    ax_tp.set_xlabel("time (s)")
    ax_tp.set_ylim(bottom=0)
    ax_kbps.set_ylim(bottom=0)

    # --- Panel 2: timings -- end-to-end latency + server encode + client decode, all ms.
    # Single run: a fixed color per metric reads clearest. Multiple runs: the run color groups a
    # file's curves and the linestyle distinguishes the metric.
    for df in runs:
        c = run_color[df.attrs["label"]]
        for col, name, lw, ls, fixed in TIMING_SERIES:
            if col not in df.columns:
                continue
            color = fixed if single else c
            lbl = name if single else f"{df.attrs['label']} {name}"
            ax_lat.plot(df["t"], df[col], color=color, lw=lw, ls=ls, label=lbl)
    ax_lat.set_ylabel("time (ms)")
    ax_lat.set_xlabel("time (s)")
    ax_lat.set_title("Timings: latency (mean) + encode + decode")
    ax_lat.set_ylim(bottom=0)

    # Phase shading only makes sense for a single run (one timeline); the phase name is written
    # in-band rather than in a legend.
    for ax in (ax_tp, ax_lat):
        if single:
            shade_phases(ax, runs[0])
            label_phases(ax, runs[0])
        ax.grid(True, alpha=0.3)

    # fps and bitrate live on separate axes (ax_tp / its twin), so merge both axes' handles into
    # one legend or the bitrate curve is left out.
    tp_h, tp_l = ax_tp.get_legend_handles_labels()
    kb_h, kb_l = ax_kbps.get_legend_handles_labels()
    if single:
        # Keep the small curve legends low so the top of each band stays clear for its label.
        ax_tp.legend(tp_h + kb_h, tp_l + kb_l, loc="lower left", fontsize=8, ncol=1)
        ax_lat.legend(loc="lower left", fontsize=8, ncol=1)
    else:
        ax_tp.legend(tp_h + kb_h, tp_l + kb_l, loc="upper left", fontsize=8, ncol=1)
        ax_lat.legend(loc="upper left", fontsize=7, ncol=1)

    if title is None:
        what = runs[0].attrs["label"] if single else f"{len(runs)} runs compared"
        title = f"Remote-rendering benchmark -- {what}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    if show:
        plt.show()


# Columns summarised in the table: (csv column, header, aggregation).
TABLE = [
    ("fps",         "fps",       "mean"),
    ("kbps",        "kbps",      "mean"),
    ("server_ms",   "server_ms", "mean"),
    ("decode_ms",   "decode_ms", "mean"),
    ("lat_mean_ms", "lat_mean",  "mean"),
    ("lat_p95_ms",  "lat_p95",   "mean"),
    ("lat_max_ms",  "lat_max",   "max"),
    ("lost",        "lost",      "sum"),
    ("ctrl_events", "ctrl",      "sum"),
]


def agg(df, cols=None):
    """Aggregate the table metrics over df (optionally a phase-filtered slice)."""
    out = {}
    for col, head, how in TABLE:
        if col not in df.columns:
            continue
        s = df[col]
        out[head] = s.mean() if how == "mean" else (s.sum() if how == "sum" else s.max())
    return out


def print_table(runs):
    heads = [h for _, h, _ in TABLE]

    def fmt(v, head):
        if head in ("lost", "ctrl"):
            return f"{int(round(v)):>10d}"
        return f"{v:>10.2f}"

    def row(name, d):
        return f"{name:<14}" + "".join(fmt(d[h], h) for h in heads if h in d)

    header = f"{'':<14}" + "".join(f"{h:>10}" for h in heads)

    if len(runs) == 1:
        df = runs[0]
        print("\nPer-phase averages (" + df.attrs["label"] + "):")
        print(header)
        print("-" * len(header))
        if "phase" in df.columns:
            # Preserve first-seen phase order rather than alphabetical.
            for ph in df["phase"].drop_duplicates():
                if ph == "done":
                    continue
                print(row(PHASE_LABELS.get(str(ph), str(ph)), agg(df[df["phase"] == ph])))
        print("-" * len(header))
        print(row("OVERALL", agg(df)))
    else:
        print("\nPer-run averages:")
        print(header)
        print("-" * len(header))
        for df in runs:
            print(row(df.attrs["label"][:14], agg(df)))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="benchmark CSV file(s) from rr-client --benchmark")
    p.add_argument("-o", "--out", help="save the figure to this path (e.g. run.png)")
    p.add_argument("-t", "--title", help="figure title (default: derived from the file name)")
    p.add_argument("--no-show", action="store_true", help="don't open a plot window")
    args = p.parse_args()

    runs = []
    for path in args.csv:
        if not Path(path).exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        runs.append(load(path))

    print_table(runs)
    plot(runs, args.out, show=not args.no_show, title=args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
