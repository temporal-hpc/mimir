#!/usr/bin/env bash
# Sweep the path-tracing workload knobs and collect one CSV row per configuration.
#
# Each run renders the live k-modal simulation under RenderPath::PathTraced with a scripted
# auto-orbit camera (input-free, reproducible) for ITERS steps, then the benchmark prints one CSV
# row to stdout (the aligned human table + banner go to stderr). We append those rows, prefixed by
# a single header, to $OUT. The tlas_time / trace_time columns come from the engine's GPU
# timestamps; spp/bounces/subdiv record the workload.
#
# Everything is overridable via environment variables, e.g.:
#   NS="1000000" SPPS="1 4 16" BOUNCES="4" ITERS=1000 ./sweep_pathtracing.sh
#
# Note: this is an on-screen rendering benchmark, so each configuration briefly opens a window.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$here/build/bin/benchmark_mimir}"
OUT="${OUT:-$here/pathtracing_sweep.csv}"

W="${W:-1920}"; H="${H:-1080}"    # window resolution (1920x1080 tags as FHD in the CSV)
SEED="${SEED:-42}"
ITERS="${ITERS:-600}"             # simulation/render steps per configuration
ORBIT="${ORBIT:-30}"              # auto-orbit deg/s (reproducible, input-free camera)

# Parameter grids (space-separated; override via env).
NS="${NS:-100000 500000 1000000}"
SPPS="${SPPS:-1 2 4}"
BOUNCES="${BOUNCES:-1 2 4}"
SUBDIVS="${SUBDIVS:-1 2}"

if [[ ! -x "$BIN" ]]; then
    echo "benchmark binary not found: $BIN" >&2
    echo "build it with: cmake --build build --target benchmark_mimir" >&2
    exit 1
fi

HEADER="mode,windowres,N,iters,seed,k,epsilon,framerate_fps,compute_time_s,pipeline_time_s,graphics_time_s,vk_usage_gb,vk_budget_gb,gpu_power_w,gpu_energy_j,gpu_time_s,nvml_free_gb,nvml_reserved_gb,nvml_total_gb,nvml_used_gb,pack_time_s,d2h_time_s,h2h_time_s,spp,bounces,subdiv,tlas_time_s,trace_time_s"
echo "$HEADER" > "$OUT"

total=0
for N in $NS; do for SPP in $SPPS; do for B in $BOUNCES; do for SD in $SUBDIVS; do
    total=$((total + 1))
    echo ">> [$total] N=$N spp=$SPP bounces=$B subdiv=$SD" >&2
    row="$(SPDLOG_LEVEL=err "$BIN" "$W" "$H" "$N" "$SEED" "$ITERS" \
        --render-path path-traced --spp "$SPP" --bounces "$B" --subdiv "$SD" \
        --orbit-speed "$ORBIT" 2>/dev/null | tail -1)"
    if [[ -n "$row" ]]; then
        echo "$row" >> "$OUT"
    else
        echo "   (no CSV row — run failed or produced no output)" >&2
    fi
done; done; done; done

echo "Wrote $total rows to $OUT" >&2
