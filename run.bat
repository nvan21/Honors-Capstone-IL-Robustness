@echo off
REM Define Conda environment name
set "CONDA_ENV=honors-capstone"

REM Define variables for each flag
set "BUFFER=./buffers/Hopper-v5/size1000000_std0.01_prand0.0.pth"
set "ALGO=airl"
set "ENV_ID=Hopper-v5"
set "NUM_STEPS=1000000"
set "EVAL_INTERVAL=50000"
set "ROLLOUT_LENGTH=20000"

REM Check if CUDA flag is needed
set "CUDA=--cuda"

REM Activate Conda environment
CALL conda activate %CONDA_ENV%

REM Run the command with the defined variables
python train_irl.py ^
    --buffer "%BUFFER%" ^
    --algo "%ALGO%" ^
    %CUDA% ^
    --env_id "%ENV_ID%" ^
    --num_steps "%NUM_STEPS%" ^
    --eval_interval "%EVAL_INTERVAL%" ^
    --rollout_length "%ROLLOUT_LENGTH%"

REM Deactivate Conda environment after execution
CALL conda deactivate
