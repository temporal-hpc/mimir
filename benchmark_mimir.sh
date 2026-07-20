#!/bin/bash

folder=$1
width=$2
height=$3
niter=$4
sync=$5
seed=$RANDOM
density=0.07

if [${sync} = 1]; then
    mode="sync"
else
    mode="desync"
fi

echo "Running mimir CA benchmark"
file=${folder}/ca_mimir_${mode}.csv
echo "mode,windowres,grid_w,grid_h,iters,seed,density,framerate_fps,compute_time_s,pipeline_time_s,graphics_time_s,vk_usage_gb,vk_budget_gb,gpu_power_w,gpu_energy_j,gpu_time_s,nvml_free_gb,nvml_reserved_gb,nvml_total_gb,nvml_used_gb,pack_time_s,d2h_time_s,h2h_time_s" >> ${file}
for exp in {6..15}; do
    size=$((2**${exp}))
    echo "    mode: ${mode}; size: ${size}"
    ./samples/ca/build/bin/benchmark_mimir ${height} ${height} ${size} ${size} ${seed} ${density} ${niter} --interop-sync ${sync} >> ${file}
done

echo "Running mimir nbody benchmark"
sizes=(10000 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000)
file=${folder}/nbody_mimir_${mode}.csv
echo "mode,windowres,N,iters,framerate_fps,compute_time_s,pipeline_time_s,graphics_time_s,vk_usage_gb,vk_budget_gb,gpu_power_w,gpu_energy_j,gpu_time_s,nvml_free_gb,nvml_reserved_gb,nvml_total_gb,nvml_used_gb,pack_time_s,d2h_time_s,h2h_time_s" >> ${file}
for size in ${sizes[@]}; do
    echo "    mode: ${mode}; size: ${size}"
    ./samples/nbody/build/bin/benchmark ${width} ${height} ${size} ${niter} ${opts} --interop-sync ${sync} >> ${file}
done

echo "Running mimir particles benchmark"
kmodal_opts="--pcolor 1.0,0.05,0.05 --background 1.0 --k 64 --epsilon 0.07 --fly --spp 1 --size 3"
renders=(flat phong ptrace)
options=("" "--light-model phong" "--light-model path-tracing")
for i in {0..2}; do
    rend=${renders[${i}]}
    opts=${options[${i}]}
    file=${folder}/kmodal_mimir_${mode}_${render}.csv
    echo "mode,windowres,N,iters,seed,k,epsilon,framerate_fps,compute_time_s,pipeline_time_s,graphics_time_s,vk_usage_gb,vk_budget_gb,gpu_power_w,gpu_energy_j,gpu_time_s,nvml_free_gb,nvml_reserved_gb,nvml_total_gb,nvml_used_gb,pack_time_s,d2h_time_s,h2h_time_s,spp,bounces,subdiv,tlas_time_s,trace_time_s" >> ${file}
    for exp in {4..8}; do
        size=$((10**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/particles-kmodal-3d/build/bin/benchmark_mimir ${width} ${height} ${size} ${seed} ${niter} ${kmodal_opts} ${opts} >> ${file}
    done
done