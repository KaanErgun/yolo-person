#!/bin/bash

# GPU Monitoring Script for Training
# Shows GPU usage, temperature, and memory in real-time

echo "🖥️  GPU Monitoring Started"
echo "Press Ctrl+C to stop"
echo ""

watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F"," "{printf \"Time: %s\nGPU: %s\nTemp: %s°C\nGPU Usage: %s%%\nMemory Usage: %s%%\nVRAM: %s/%s MB\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7}"'
