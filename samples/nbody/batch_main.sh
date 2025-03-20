#!/bin/bash

iters=1000
present=0
sizes=(1000000)
fps=(100)
widths=(1920)
heights=(1080)
syncs=(1)
echo "mode,windowres,N,target_fps,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used" >> $1
for i in ${!widths[@]}; do
    w=${widths[$i]}
    h=${heights[$i]}
    viewport="${w}x${h}"
    #nvidia-settings -a CurrentMetaMode="DPY-3: 1920x1080_144 @1920x1080 +1920+0
    #    {ViewPortIn=${viewport}, ViewPortOut=${viewport}+0+0}, DPY-2: nvidia-auto-select
    #    @1920x1080 +0+0 {ViewPortIn=1920x1080, ViewPortOut=1920x1080+0+0}"
    echo "Viewport: " ${viewport}
    for sync in ${syncs[@]}; do
        echo "  Sync mode: ${sync}"
        for target in ${fps[@]}; do
            echo "    Target FPS: ${target}"
            for n in ${sizes[@]}; do
                echo "      Size: ${n}"
                ./cudaview/build/samples/benchmark ${w} ${h} ${n} ${iters} ${present} ${target} ${sync} >> $1
            done
        done
    done
done

