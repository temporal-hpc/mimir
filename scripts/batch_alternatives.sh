#!/bin/bash

sizes=(1000 10000 100000 1000000 10000000)
iters=1000
present=2
fps=0
sync=1

echo "mode,N,framerate,freemem,usedmem,power,energy,time" >> $1
for n in ${sizes[@]}; do
    ./cudaview/build/samples/pointsgl/pointsgl 1920 1080 ${n} ${iters} >> $1
done
for n in ${sizes[@]}; do
    ./cudaview/build/samples/points3d 1920 1080 ${n} ${iters} ${present} ${fps} ${sync} 1 >> $1
done
for n in ${sizes[@]}; do
    ./cudaview/build/samples/pointspcl/pointspcl ${n} ${iters} >> $1
done