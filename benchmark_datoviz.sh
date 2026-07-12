#!/bin/bash

folder=$1
width=$2
height=$3
niter=$4
seed=$RANDOM

echo "Running datoviz CA benchmark"
file=${folder}/ca_datoviz.csv
echo "mode,windowres,grid_w,grid_h,seed,density,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
for exp in {6..15}; do
    size=$((2**${exp}))
    echo "    size: ${size}"
    ./samples/ca/build/bin/benchmark_datoviz ${height} ${height} ${size} ${size} ${seed} 0.07 ${niter} >> ${file}
done

sizes=(10000 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000)
echo "Running datoviz nbody benchmark"
file=${folder}/nbody_datoviz.csv
echo "mode,windowres,N,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
for size in ${sizes[@]}; do
    echo "    size: ${size}"
    ./samples/nbody-datoviz/build/bin/benchmark_datoviz ${width} ${height} ${size} ${niter} >> ${file}
done

modes=(simple phong)
options=("" "--light-model phong")
kmodal_common="--pcolor 1.0,0.05,0.05 --background 1.0 --k 64 --epsilon 0.07 --fly --size 3"
echo "Running datoviz particles benchmark"
for i in {0..1}; do
    mode=${modes[${i}]}
    opts=${options[${i}]}
    file=${folder}/kmodal_datoviz_${mode}.csv
    echo "mode,windowres,N,seed,k,epsilon,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for exp in {4..8}; do
        size=$((10**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/particles-kmodal-3d/build/bin/benchmark_datoviz ${width} ${height} ${size} ${seed} ${niter} ${kmodal_common} ${opts} >> ${file}
    done
done
