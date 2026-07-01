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
echo "mode,windowres,N,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used" >> $1
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

