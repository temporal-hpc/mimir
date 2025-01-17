#!/bin/bash

sizes=(1000 10000 100000 1000000 10000000 100000000)
iters=1000
present=2
fps=0
sync=0

echo "sync_mode,N,windowres,target_fps,framerate,compute_time,pipeline_time,graphics_time,usage,budget,freemem,usedmem,power,energy,time" >> $1
for n in ${sizes[@]}; do
    ./cudaview/build/samples/points3d 0 0 ${n} ${iters} ${present} ${fps} ${sync} 1 >> $1
done
for n in ${sizes[@]}; do
    ./cudaview/build/samples/points3d 0 0 ${n} ${iters} ${present} ${fps} ${sync} 0 >> $1
done
for n in ${sizes[@]}; do
    ./cudaview/build/samples/points3dcpu 0 0 ${n} ${iters} ${present} ${fps} ${sync} >> $1
done
