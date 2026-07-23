#!/usr/bin/env python3
"""Generate a small LaTeX table listing the four benchmarked GPUs and their client/server
locations (client is always Chile; see research/data/server-countries for the server side).

Reuses plot_grid.py's GPUS list as the single source of truth for device name, memory, and
server country, so this table and the figures' legends never drift apart.

The table is wrapped in resizebox{columnwidth}{!}{...} so it shrinks to fit a two-column
paper's column width regardless of how long the device/country names get -- needs the
graphicx package.

Usage:
    table_gpus.py                            # -> research/tables/gpus.tex
    table_gpus.py -o research/tables/gpus.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_grid as pg  # noqa: E402

CLIENT_COUNTRY = "CL"
COUNTRY_NAME = {"CL": "Chile", "RO": "Romania", "FR": "France", "NL": "Netherlands"}


def to_latex():
    rows = [(device, mem, country) for _, _, device, mem, country, _ in pg.GPUS]
    name_w = max(len(device) for device, _, _ in rows)
    mem_w = max(len(mem) for _, mem, _ in rows)
    client = f"{COUNTRY_NAME[CLIENT_COUNTRY]} ({CLIENT_COUNTRY})"
    loc_w = max(len(client), max(len(f"{COUNTRY_NAME[c]} ({c})") for _, _, c in rows))

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{GPUs benchmarked and their client/server locations.}",
        r"  \label{tab:rr-benchmark-gpus}",
        r"  \resizebox{\columnwidth}{!}{%",
        r"  \begin{tabular}{lrll}",
        r"    \toprule",
        r"    GPU & Memory & Client & Server \\",
        r"    \midrule",
    ]
    for device, mem, country in rows:
        server = f"{COUNTRY_NAME[country]} ({country})"
        lines.append(f"    {device:<{name_w}} & {mem:>{mem_w}} & "
                      f"{client:<{loc_w}} & {server:<{loc_w}} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    repo = here.parent.parent  # research/scripts -> research -> repo
    p.add_argument("-o", "--out", default=str(repo / "research" / "tables" / "gpus.tex"),
                   help="output .tex path (default: research/tables/gpus.tex)")
    args = p.parse_args()

    tex = to_latex()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
