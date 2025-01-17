#!/bin/bash

iters=10000
present=0
sizes=(1000000 10000000 100000000)
fps=(10)
widths=(1920)
heights=(1080)
syncs=(0 1)
echo "sync_mode,N,windowres,target_fps,framerate,compute_time,pipeline_time,graphics_time,usage,budget,freemem,usedmem,power,energy,time" >> $1
for i in ${!widths[@]}; do
    w=${widths[$i]}
    h=${heights[$i]}
    viewport="${w}x${h}"
    nvidia-settings -a CurrentMetaMode="DPY-3: 1920x1080_144 @1920x1080 +1920+0 
        {ViewPortIn=${viewport}, ViewPortOut=${viewport}+0+0}, DPY-2: nvidia-auto-select
        @1920x1080 +0+0 {ViewPortIn=1920x1080, ViewPortOut=1920x1080+0+0}"
    echo "Viewport: " ${viewport}
    for sync in ${syncs[@]}; do
        echo "  Sync mode: ${sync}"
        for target in ${fps[@]}; do
            echo "    Target FPS: ${target}"
            for n in ${sizes[@]}; do
                echo "      Size: ${n}"
                ./cudaview/build/samples/points3d ${w} ${h} ${n} ${iters} ${present} ${target} ${sync} >> $1
            done
        done
    done
done

