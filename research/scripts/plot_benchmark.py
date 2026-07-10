#!/usr/bin/env python3
"""Plot the remote-rendering benchmark time series and print a summary table.

Consumes the CSV written by ``rr-client --benchmark F`` (see rr-client.cpp). Columns:

    time_s, fps, kbps, server_ms, decode_ms,
    lat_mean_ms, lat_p50_ms, lat_p95_ms, lat_max_ms, lost, ctrl_events, phase

Only the metrics worth watching *over time* get a curve; the rest live in a table of
per-phase (and overall) averages. The four time-series panels group curves that share a
Y-axis magnitude, or split onto twin axes when they do not:

    1. Throughput      fps (left axis)  +  bitrate kbps (right axis)   -- twin Y
    2. End-to-end lat. lat_mean / p50 / p95 / max                     -- shared ms
    3. Frame pipeline  server_ms (encode/readback) + decode_ms        -- shared ms
    4. Frame loss      lost per stats window                          -- count

`server_ms` is the server's per-frame production cost: NVENC *encode* time for an H.264
stream, or framebuffer *readback* time for a raw stream (see rr-client.cpp:588).

Usage:
    plot_benchmark.py run.csv                       # one run: panels are phase-shaded
    plot_benchmark.py a.csv b.csv c.csv             # compare runs: overlaid, legend by file
    plot_benchmark.py run.csv -o run.png            # save instead of (also) showing
    plot_benchmark.py run.csv --no-show             # table only, no window

Requires: pandas, matplotlib (pip install pandas matplotlib).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Metrics that get a time-series curve are defined by the panels below; everything else in
# the CSV is summarised in the table only. Phase spans are shaded with these colors.
PHASE_COLORS = {
    "idle":     "#d9d9d9",
    "orbit":    "#c6dbef",
    "zoom_in":  "#c7e9c0",
    "inside":   "#fdd0a2",
    "zoom_out": "#dadaeb",
    "done":     "#f0f0f0",
}


def load(path):
    """Read one benchmark CSV, normalise time to start at 0, tag it with a short label."""
    df = pd.read_csv(path)
    df["t"] = df["time_s"] - df["time_s"].iloc[0]
    df.attrs["label"] = Path(path).stem
    return df


def shade_phases(ax, df, label=False):
    """Shade contiguous same-phase spans behind a single run's curves.

    Only pass label=True on one panel: shading labels there feed the phase legend, and
    labelling every panel would pollute each panel's own curve legend with phase entries.
    """
    if "phase" not in df.columns:
        return
    start = 0
    phase = df["phase"].iloc[0]
    seen = set()
    for i in range(1, len(df) + 1):
        cur = df["phase"].iloc[i] if i < len(df) else None
        if cur != phase:
            color = PHASE_COLORS.get(str(phase), "#eeeeee")
            t0 = df["t"].iloc[start]
            t1 = df["t"].iloc[i - 1]
            ax.axvspan(t0, t1, color=color, alpha=0.5, lw=0,
                       label=(phase if label and phase not in seen else None), zorder=0)
            seen.add(phase)
            start, phase = i, cur


def plot(runs, out, show):
    single = len(runs) == 1
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    (ax_tp, ax_lat), (ax_pipe, ax_loss) = axes
    # A stable color per run so the same file reads the same across all panels.
    run_color = {r.attrs["label"]: c for r, c in
                 zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"])}

    # --- Panel 1: throughput -- fps and bitrate live on separate magnitudes, so twin axes.
    ax_kbps = ax_tp.twinx()
    for df in runs:
        c = run_color[df.attrs["label"]]
        lbl = df.attrs["label"]
        ax_tp.plot(df["t"], df["fps"], color=c, lw=1.6,
                   label=f"{lbl} fps" if not single else "fps")
        ax_kbps.plot(df["t"], df["kbps"], color=c, lw=1.1, ls="--",
                     label=f"{lbl} kbps" if not single else "kbps")
    ax_tp.set_ylabel("frames / s")
    ax_kbps.set_ylabel("bitrate (kbps, dashed)")
    ax_tp.set_title("Throughput: FPS (solid) + bitrate (dashed)")
    ax_tp.set_ylim(bottom=0)
    ax_kbps.set_ylim(bottom=0)

    # --- Panel 2: end-to-end latency -- all ms, shared axis.
    lat_cols = [("lat_mean_ms", "mean", 1.8, "-"),
                ("lat_p50_ms", "p50", 1.0, ":"),
                ("lat_p95_ms", "p95", 1.2, "--"),
                ("lat_max_ms", "max", 0.9, "-.")]
    for df in runs:
        c = run_color[df.attrs["label"]]
        for col, name, lw, ls in lat_cols:
            if col in df.columns:
                lbl = name if single else f"{df.attrs['label']} {name}"
                ax_lat.plot(df["t"], df[col], color=c, lw=lw, ls=ls, label=lbl)
    ax_lat.set_ylabel("latency (ms)")
    ax_lat.set_title("End-to-end latency (mean / p50 / p95 / max)")
    ax_lat.set_ylim(bottom=0)

    # --- Panel 3: per-frame pipeline -- server vs client cost, both ms, shared axis.
    for df in runs:
        c = run_color[df.attrs["label"]]
        ax_pipe.plot(df["t"], df["server_ms"], color=c, lw=1.6,
                     label="server (encode/readback)" if single else f"{df.attrs['label']} server")
        ax_pipe.plot(df["t"], df["decode_ms"], color=c, lw=1.2, ls="--",
                     label="decode" if single else f"{df.attrs['label']} decode")
    ax_pipe.set_ylabel("per-frame (ms)")
    ax_pipe.set_xlabel("time (s)")
    ax_pipe.set_title("Frame pipeline: server (solid) vs decode (dashed)")
    ax_pipe.set_ylim(bottom=0)

    # --- Panel 4: frame loss per stats window.
    for df in runs:
        c = run_color[df.attrs["label"]]
        ax_loss.plot(df["t"], df["lost"], color=c, lw=1.4,
                     label=None if single else df.attrs["label"])
    ax_loss.set_ylabel("frames lost / window")
    ax_loss.set_xlabel("time (s)")
    ax_loss.set_title("Frame loss")
    ax_loss.set_ylim(bottom=0)

    # Phase shading only makes sense for a single run (one timeline); otherwise use a legend.
    # Label the spans on the loss panel only, so it alone carries the phase legend.
    for ax in (ax_tp, ax_lat, ax_pipe, ax_loss):
        if single:
            shade_phases(ax, runs[0], label=(ax is ax_loss))
        ax.grid(True, alpha=0.3)

    if single:
        # Per-panel curve legends stay local; the loss panel shows the phase-color key.
        ax_tp.legend(loc="lower left", fontsize=8)
        ax_lat.legend(loc="upper left", fontsize=8, ncol=2)
        ax_pipe.legend(loc="upper left", fontsize=8)
        ax_loss.legend(loc="upper right", fontsize=8, title="phase", ncol=2)
    else:
        for ax in (ax_tp, ax_lat, ax_pipe):
            ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax_loss.legend(loc="upper left", fontsize=8, title="run")

    title = runs[0].attrs["label"] if single else f"{len(runs)} runs compared"
    fig.suptitle(f"Remote-rendering benchmark -- {title}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

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
                print(row(str(ph), agg(df[df["phase"] == ph])))
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
    p.add_argument("--no-show", action="store_true", help="don't open a plot window")
    args = p.parse_args()

    runs = []
    for path in args.csv:
        if not Path(path).exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        runs.append(load(path))

    print_table(runs)
    plot(runs, args.out, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
