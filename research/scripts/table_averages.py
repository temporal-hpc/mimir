#!/usr/bin/env python3
"""Generate one big LaTeX averages table from every benchmark CSV in research/data/.

One row per (GPU, mode, N) run -- 36 rows for the current dataset -- with the same
whole-run averages `plot_benchmark.py` prints to the terminal (fps, latency mean, server
encode/readback, client decode). Reuses plot_grid.py's CSV discovery (parse_meta/collect) and
plot_benchmark.py's aggregation (agg) so the table always matches what the figures show --
nothing here recomputes an average independently.

Usage:
    table_averages.py                            # -> research/tables/averages.tex
    table_averages.py --data-dir research/data -o research/tables/averages.tex

Requires: pandas (via plot_benchmark.py).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark as pb  # noqa: E402
import plot_grid as pg  # noqa: E402

# Short mode labels for a table column, vs. plot_grid's longer figure-title names.
MODE_LABEL = {"pt": "PT", "phong": "Phong", "raster": "Raster"}

# Table columns: (header, agg() key, format spec).
COLUMNS = [
    ("FPS",          "fps",       "{:.1f}"),
    ("Latency (ms)", "lat_mean",  "{:.1f}"),
    ("Encode (ms)",  "server_ms", "{:.2f}"),
    ("Decode (ms)",  "decode_ms", "{:.2f}"),
]


def build_rows(items):
    """One row per CSV, ordered GPU -> mode -> size-regime, matching the grid figures' layout.
    Uses the short GPU name (table column, not the long legend name used in the figures)."""
    rows = []
    for gpu_key, gpu_name, _legend_name, _color in pg.GPUS:
        for mode in pg.MODES:
            for col_key, size_set, _ in pg.COLUMNS:
                matches = [(m, p) for m, p in items
                           if m["gpu"] == gpu_key and m["mode"] == mode and m["size"] in size_set]
                for meta, path in matches:
                    df = pb.load(path)
                    stats = pb.agg(df)
                    rows.append((gpu_name, MODE_LABEL[mode], pg.format_n(meta["size"]), stats))
    return rows


def to_latex(rows):
    header = " & ".join(["GPU", "Mode", "N"] + [h for h, _, _ in COLUMNS])
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Remote-rendering benchmark averages (whole-run means).}",
        r"  \label{tab:rr-benchmark-averages}",
        r"  \begin{tabular}{ll" + "r" * (1 + len(COLUMNS)) + "}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    prev_gpu = None
    for gpu, mode, n, stats in rows:
        # A blank rule between GPUs, so the table reads as GPU-grouped blocks.
        if prev_gpu is not None and gpu != prev_gpu:
            lines.append(r"    \addlinespace")
        vals = " & ".join(fmt.format(stats[key]) for _, key, fmt in COLUMNS)
        lines.append(f"    {gpu} & {mode} & {n} & {vals} \\\\")
        prev_gpu = gpu
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    repo = here.parent.parent  # research/scripts -> research -> repo
    p.add_argument("--data-dir", default=str(repo / "research" / "data"),
                   help="directory of benchmark CSVs (default: research/data)")
    p.add_argument("-o", "--out", default=str(repo / "research" / "tables" / "averages.tex"),
                   help="output .tex path (default: research/tables/averages.tex)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: no such data dir: {data_dir}", file=sys.stderr)
        return 1
    items = pg.collect(data_dir)
    if not items:
        print(f"error: no benchmark CSVs matched the naming convention in {data_dir}",
              file=sys.stderr)
        return 1

    rows = build_rows(items)
    tex = to_latex(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
