#!/bin/bash
# test_dist.sh
# Run a 2-process training session locally using torchrun.

echo "Starting 2-process distributed training smoke test..."

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:.

# Use gloo backend for local CPU testing if CUDA is not available
# We pass run.steps=20 to make it quick
torchrun --nproc_per_node=2 scripts/train.py run.steps=20 run.dist_backend=gloo

if [ $? -eq 0 ]; then
    echo "Distributed training test PASSED"
else
    echo "Distributed training test FAILED"
    exit 1
fi

# Check if parquet files were generated
if [ -f "metrics_rank0.parquet" ]; then
    echo "Metrics file found: metrics_rank0.parquet"
else
    echo "Warning: metrics_rank0.parquet NOT found"
fi
