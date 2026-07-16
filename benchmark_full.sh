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
    echo "  $0 result 2560 1440 10000"
    exit 0
fi

echo "Warming up..."
./samples/nbody/build/bin/benchmark 1920 1080 65536 300

echo "Starting benchmark with window size (${2}x${3}) and ${4} iterations"
./benchmark_mimir_alt.sh $1 $2 $3 $4
#./benchmark_mimir.sh $1 $2 $3 $4
#./benchmark_datoviz.sh $1 $2 $3 $4
