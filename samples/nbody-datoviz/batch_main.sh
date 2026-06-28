#!/bin/bash
# Same driver as samples/nbody/batch_main.sh, pointing at the datoviz binary and with
# the extra transfer_time column appended to the CSV header.

if [[ "$1" == "--help" || "$1" == "-h" || -z "$1" ]]; then
    echo "Usage: $0 <output.csv>"
    echo ""
    echo "  output.csv   File to append results to (created if it does not exist)"
    echo ""
    echo "Runs benchmark_datoviz across the parameter grid defined in this script"
    echo "and appends one CSV row per configuration (plus a header) to output.csv."
    echo ""
    echo "Example:"
    echo "  $0 results.csv"
    exit 0
fi

iters=1000
present=0
sizes=(1000000)
fps=(100)
widths=(1920)
heights=(1080)
syncs=(1)
echo "mode,windowres,N,target_fps,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,transfer_time" >> $1
for i in ${!widths[@]}; do
    w=${widths[$i]}
    h=${heights[$i]}
    viewport="${w}x${h}"
    echo "Viewport: " ${viewport}
    for sync in ${syncs[@]}; do
        echo "  Sync mode: ${sync}"
        for target in ${fps[@]}; do
            echo "    Target FPS: ${target}"
            for n in ${sizes[@]}; do
                echo "      Size: ${n}"
                ./build/bin/benchmark_datoviz ${w} ${h} ${n} ${iters} ${present} ${target} ${sync} >> $1
            done
        done
    done
done
