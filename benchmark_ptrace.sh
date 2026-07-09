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

modes=(mimir)
width=1920
height=1080
niter=100
seed=$RANDOM

sizes=(10000 50000 100000 150000 200000 250000 300000 350000
       400000 450000 500000 550000 600000 650000 700000 750000
       800000 850000 900000 950000 1000000)
echo "Running particles benchmark with window size (${width}x${height}) and RNG state ${seed}"
for i in {0..1}; do
    mode=${modes[${i}]}
    file=$1_${mode}_ptrace_kmodal.csv
    echo "mode,windowres,N,seed,k,epsilon,framerate,compute_time,pipeline_time,graphics_time,vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,nvml_used,pack_time,d2h_time,h2h_time" >> ${file}
    for exp in {4..8}; do
        size=$((10**${exp}))
        echo "    mode: ${mode}; size: ${size}"
        ./samples/particles-kmodal-3d/build/bin/benchmark_${mode} ${width} ${height} ${size} 413111 ${niter} --pcolor 1.0,0.05,0.05 --background 1.0 --k 64 --epsilon 0.07 --fly --light-model path-tracing --spp 1 --size 3 >> ${file}
    done
done