#!/bin/bash

# Define variables for each flag
BUFFER="buffers/Hopper-v5/size1000000_std0.01_prand0.0.pth"
ALGO="airl"
CUDA="--cuda"  # Leave empty if you don't want to use CUDA
ENV_ID="Hopper-v5"
NUM_STEPS=1000000
EVAL_INTERVAL=50000
ROLLOUT_LENGTH=20000

# Run the command with the defined variables
python train_irl.py \
    --buffer "$BUFFER" \
    --algo "$ALGO" \
    $CUDA \
    --env_id "$ENV_ID" \
    --num_steps "$NUM_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --rollout_length "$ROLLOUT_LENGTH"
