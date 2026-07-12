#!/bin/bash

folder=$1
width=$2
height=$3
niter=$4
seed=$RANDOM
density=0.07

modes=(sync desync)
options=("" "--interop-sync 0")

echo "Running mimir CA benchmark"
for mode in ${modes[@]}; do
    file=${folder}/ca_mimir_${mode}.csv
    echo "mode,windowres,grid_w,grid_h,seed,density,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for exp in {6..15}; do
        size=$((2**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/ca/build/bin/benchmark_mimir ${height} ${height} ${size} ${size} ${seed} ${density} ${niter} >> ${file}
    done
done

sizes=(10000 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000)
echo "Running mimir nbody benchmark"
for i in {0..1}; do
    mode=${modes[${i}]}
    opts=${options[${i}]}
    file=${folder}/nbody_mimir_${mode}.csv
    echo "mode,windowres,N,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for size in ${sizes[@]}; do
        echo "    mode: ${mode}; size: ${size}"
        ./samples/nbody/build/bin/benchmark ${width} ${height} ${size} ${niter} ${opts} >> ${file}
    done
done

echo "Running mimir particles benchmark"
kmodal_common="--pcolor 1.0,0.05,0.05 --background 1.0 --k 64 --epsilon 0.07 --fly --spp 1 --size 3"
for i in {0..1}; do
    mode=${modes[${i}]}
    opts=${options[${i}]}
    file=${folder}/kmodal_mimir_${mode}.csv
    echo "mode,windowres,N,seed,k,epsilon,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for exp in {4..8}; do
        size=$((10**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/particles-kmodal-3d/build/bin/benchmark_mimir ${width} ${height} ${size} ${seed} ${niter} ${kmodal_common} ${opts} >> ${file}
    done
done

modes=(phong ptrace)
options=("--light-model phong" "--light-model path-tracing") #TODO
echo "Running mimir particles benchmark (alternate rendering modes)"
for i in {0..1}; do
    mode=${modes[${i}]}
    opts=${options[${i}]}
    file=${folder}/kmodal_mimir_sync_${mode}.csv
    echo "mode,windowres,N,seed,k,epsilon,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for exp in {4..8}; do
        size=$((10**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/particles-kmodal-3d/build/bin/benchmark_mimir ${width} ${height} ${size} ${seed} ${niter} ${kmodal_common} ${opts} >> ${file}
    done
done