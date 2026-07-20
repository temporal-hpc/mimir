#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" || -z "$1" ]]; then
    echo "Usage: $0 <output.csv>"
    echo ""
    echo "  output.csv   File to append results to (created if it does not exist)"
    echo ""
    echo "Runs benchmark across the parameter grid defined in this script"
    echo "and appends one CSV row per configuration (plus a header) to output.csv."
    echo ""
    echo "Example:"
    echo "  $0 results.csv"
    exit 0
fi

iters=1000
present=0
sizes=(1000000)
widths=(1920)
heights=(1080)
syncs=(1)
echo "mode,windowres,N,iters,framerate_fps,compute_time_s,pipeline_time_s,graphics_time_s,vk_usage_gb,vk_budget_gb,gpu_power_w,gpu_energy_j,gpu_time_s,nvml_free_gb,nvml_reserved_gb,nvml_total_gb,nvml_used_gb,pack_time_s,d2h_time_s,h2h_time_s" >> $1
for i in ${!widths[@]}; do
    w=${widths[$i]}
    h=${heights[$i]}
    viewport="${w}x${h}"
    #nvidia-settings -a CurrentMetaMode="DPY-3: 1920x1080_144 @1920x1080 +1920+0
    #    {ViewPortIn=${viewport}, ViewPortOut=${viewport}+0+0}, DPY-2: nvidia-auto-select
    #    @1920x1080 +0+0 {ViewPortIn=1920x1080, ViewPortOut=1920x1080+0+0}"
    echo "Viewport: " ${viewport}
    for sync in ${syncs[@]}; do
        echo "  Interop-sync: ${sync}"
        for n in ${sizes[@]}; do
            echo "    Size: ${n}"
            ./build/bin/benchmark ${w} ${h} ${n} ${iters} --present ${present} --interop-sync ${sync} >> $1
        done
    done
done

