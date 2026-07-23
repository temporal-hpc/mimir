#!/usr/bin/env python3
"""Plot the remote-rendering benchmark time series and print a summary table.

Consumes the CSV written by ``rr-client --benchmark F`` (see rr-client.cpp). Columns:

    time_s, fps, kbps, server_ms, server_ms_std, decode_ms, decode_ms_std,
    lat_mean_ms, lat_std_ms, lat_p50_ms, lat_p95_ms, lat_max_ms, lost, ctrl_events, phase

Only the metrics worth watching *over time* get a curve; the rest live in a table of
per-phase (and overall) averages. Two time-series panels:

    1. Throughput   fps
    2. Latency (ms) mean end-to-end latency, with a +/-1 std-dev band

The *_std columns are per-window std-devs (feeding the error bands). Everything else
(bitrate, server encode/readback, client decode, latency p50/p95/max, frame loss, control
events) is reported in the table only, not plotted. `server_ms` is the server's per-frame
production cost: NVENC *encode* time for an H.264 stream, or framebuffer *readback* time for
a raw stream.

Usage:
    plot_benchmark.py run.csv                       # one run: panels are phase-shaded
    plot_benchmark.py a.csv b.csv c.csv             # compare runs: overlaid, legend by file
    plot_benchmark.py run.csv -o run.png            # save instead of (also) showing
    plot_benchmark.py run.csv -t "My title"         # override the figure title
    plot_benchmark.py run.csv --logy                # log-scale the timings (ms) panel
    plot_benchmark.py run.csv --yrange-a 0 65 --yrange-b 0 80        # fix axis ranges
    plot_benchmark.py run.csv --no-show             # table only, no window
    # One PDF per panel with a use-case subtitle (for a use-cases x metric-rows grid):
    plot_benchmark.py cloud.csv --split -o cloud.pdf -s "Cloud service"
        #  -> cloud-throughput.pdf and cloud-timings.pdf

Requires: pandas, matplotlib (pip install pandas matplotlib).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Global font sizing: these figures print at full page width (esp. the plot_grid.py 2x3
# grids), so the default matplotlib sizes read as too small on paper. Bumped up globally
# rather than per-call so plot_grid.py's imported draw_* calls pick it up too. The legend gets
# a little transparency (framealpha) so it doesn't fully block curves/shading underneath it.
plt.rcParams.update({
    "font.size": 23,
    "axes.titlesize": 27,
    "axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.framealpha": 0.55,
})

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
    "far":         "Far view (Static)",
    "orbit":       "Orbit view (Mid Dynamics)",
    "zoom_in":     "Zoom in (Low Dynamics)",
    "look_around": "Look Around (High Dynamics)",
    "inside":      "Inside view (Static)",
    # Legacy tokens from earlier CSVs.
    "outside":     "Far view (Static)",
    "idle":        "Far view (Static)",
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
                        fontsize=11, style="italic", color="#666666", alpha=0.7,
                        linespacing=1.4, zorder=1)
            start, phase = i, cur


# Timings panel: mean end-to-end latency with a +/-1 std-dev error band. Encode/decode are
# stable enough to skip on the plot -- they, plus latency percentiles (p50/p95) and max, stay
# in the averages table instead.
# (value column, linewidth, linestyle, std-dev column).
TIMING_SERIES = [
    ("lat_mean_ms", 1.8, "-", "lat_std_ms"),
]


def _top(runs, cols, std_map=None, frac=0.16):
    """Data max across runs for `cols` (adding each column's std where given), plus `frac`
    headroom so a curve that just touches its peak doesn't merge into the top border."""
    m = 0.0
    for df in runs:
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col]
            std = (std_map or {}).get(col)
            if std and std in df.columns:
                s = s + df[std]
            m = max(m, float(s.max()))
    return m * (1.0 + frac) if m > 0 else None


# Timing value column -> its std column, so the headroom calc clears the error bands too.
TIMING_STD = {col: stdcol for col, _, _, stdcol in TIMING_SERIES}


def draw_throughput(ax, runs, run_color, single, yrange_a):
    """Draw the throughput panel (fps) into a caller-owned axes."""
    # Each run contributes exactly one curve here, so the legend is just the run's label --
    # no metric suffix needed, it's already named by the panel title / y-axis.
    for df in runs:
        lbl = df.attrs["label"]
        c = run_color[lbl]
        ax.plot(df["t"], df["fps"], color=c, lw=1.6, label=lbl)
    ax.set_ylabel("FPS")
    ax.set_title("Throughput: FPS", pad=18)
    ax.set_xlabel("time (s)")
    if yrange_a:
        ax.set_ylim(yrange_a[0], yrange_a[1])
    else:
        ax.set_ylim(0, _top(runs, ["fps"]))
    # Phase boundaries are time-driven, not frame-driven, so they land within well under a
    # second of each other across runs -- shading against runs[0] reads fine even overlaid.
    shade_phases(ax, runs[0])
    label_phases(ax, runs[0])
    ax.grid(True, alpha=0.3)
    # borderaxespad gives the box a visible gap from the axes spines, so its corner doesn't
    # read as clipped by the frame. Monospace so plot_grid.py's space-padded GPU labels (GB
    # and country code columns) actually line up -- harmless for plain filename labels too.
    # Font size pinned below rcParams' legend.fontsize -- this legend carries a lot of aligned
    # text (GPU name + memory + location), so it doesn't need to scale with the rest of the
    # figure's much bigger fonts or it swallows the plot area / collides with the axes.
    ax.legend(loc="lower left", ncol=1, borderaxespad=1.0,
              prop={"family": "monospace", "size": 16})


def draw_timings(ax, runs, run_color, single, yrange_b, logy):
    """Draw the timings panel (mean latency, with a std band) into `ax`."""
    # One curve per run (latency only), same legend convention as draw_throughput: just the
    # run's label, no metric suffix.
    for df in runs:
        lbl = df.attrs["label"]
        c = run_color[lbl]
        for col, lw, ls, stdcol in TIMING_SERIES:
            if col not in df.columns:
                continue
            ax.plot(df["t"], df[col], color=c, lw=lw, ls=ls, label=lbl)
            # +/-1 std-dev error band, shaded in the same color (skipped for CSVs without the
            # *_std columns). zorder keeps it above phase shading but below the curves.
            if stdcol and stdcol in df.columns:
                lo = (df[col] - df[stdcol]).clip(lower=0)
                hi = df[col] + df[stdcol]
                ax.fill_between(df["t"], lo, hi, color=c, alpha=0.15, lw=0, zorder=0.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("time (s)")
    ax.set_title("Latency", pad=18)
    if logy:
        ax.set_yscale("log")
    if yrange_b:
        ax.set_ylim(yrange_b[0], yrange_b[1])
    elif not logy:  # log autoscales to positive data; a 0 floor would be invalid
        ax.set_ylim(0, _top(runs, list(TIMING_STD), std_map=TIMING_STD))
    # Shading only here (no text): when stacked under the throughput panel the phase names
    # already sit up top, and repeating them would just clutter this row.
    shade_phases(ax, runs[0])
    if single:
        label_phases(ax, runs[0])
    ax.grid(True, alpha=0.3)
    # Upper right is free to use only when this panel has no phase-name text up top (i.e. not
    # single -- see label_phases above); single-run mode keeps it low to stay clear of that text.
    ax.legend(loc="lower right" if single else "upper right", ncol=1, borderaxespad=1.0,
              prop={"family": "monospace", "size": 16})


def _figtitles(fig, title, subtitle):
    """Figure title, with an optional smaller italic subtitle beneath it (e.g. the use case)."""
    fig.suptitle(title, fontsize=27, y=0.98)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=20,
                 style="italic", color="#444444")


def plot(runs, out, show, title=None, subtitle=None, yrange_a=None, yrange_b=None,
         logy=False, split=False):
    single = len(runs) == 1
    # A stable color per run so the same file reads the same across both panels.
    run_color = {r.attrs["label"]: c for r, c in
                 zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    if title is None:
        title = "Mìmir Remote Rendering Benchmark"
        if subtitle is None:
            subtitle = runs[0].attrs["label"] if single else f"{len(runs)} runs compared"
    top = 0.90 if subtitle else 0.96

    if split:
        # One figure (and one output file) per panel, for assembling an external grid of use
        # cases x metric rows. `-o base.pdf` -> base-throughput.pdf and base-timings.pdf.
        panels = [("throughput", draw_throughput, (run_color, single, yrange_a)),
                  ("timings",    draw_timings,    (run_color, single, yrange_b, logy))]
        for suffix, draw, extra in panels:
            fig, ax = plt.subplots(figsize=(7, 5))
            draw(ax, runs, *extra)
            _figtitles(fig, title, subtitle)
            fig.tight_layout(rect=(0, 0, 1, top))
            if out:
                base = Path(out)
                path = base.with_name(f"{base.stem}-{suffix}{base.suffix or '.pdf'}")
                fig.savefig(path, dpi=130, bbox_inches="tight")
                print(f"wrote {path}")
        if show:
            plt.show()
        return

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    draw_throughput(ax_tp, runs, run_color, single, yrange_a)
    draw_timings(ax_lat, runs, run_color, single, yrange_b, logy)
    _figtitles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, top))
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
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
    p.add_argument("-o", "--out", help="save the figure to this path (e.g. run.pdf). With "
                   "--split this is a base name: base-throughput.pdf and base-timings.pdf")
    p.add_argument("-t", "--title", help="figure title (default: derived from the file name)")
    p.add_argument("-s", "--subtitle", help="smaller line under the title, e.g. the use case "
                   "(\"Office workstation\", \"University cluster\", \"Cloud service\")")
    p.add_argument("--split", action="store_true",
                   help="save each panel as its own file (for assembling an external grid); "
                        "use with -o to pick the base name/format (defaults to PDF)")
    p.add_argument("--yrange-a", nargs=2, type=float, metavar=("LOW_FPS", "HIGH_FPS"),
                   help="fix the throughput (fps) panel y-axis: low high")
    p.add_argument("--yrange-b", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="fix the timings (ms) panel y-axis: low high")
    p.add_argument("--logy", action="store_true",
                   help="log-scale the timings (ms) panel y-axis")
    p.add_argument("--no-show", action="store_true", help="don't open a plot window")
    args = p.parse_args()

    runs = []
    for path in args.csv:
        if not Path(path).exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        runs.append(load(path))

    print_table(runs)
    plot(runs, args.out, show=not args.no_show, title=args.title, subtitle=args.subtitle,
         yrange_a=args.yrange_a, yrange_b=args.yrange_b, logy=args.logy, split=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
