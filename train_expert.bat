@echo off
REM Define Conda environment name
set "CONDA_ENV=honors-capstone"

REM Define variables for each flag
set "REWARD_MODEL=./logs/Hopper-v5/airl_ppo/seed0-20250325-1534/model/step5000000/disc.pth"
set "MODIFIED_XML=./xml/Hopper-v5/decreased_friction.xml"
set "ENV_ID=Hopper-v5"
set "NUM_STEPS=1000000"
set "EVAL_INTERVAL=50000"
set "SEED=0"

REM Check if CUDA flag is needed
set "CUDA=--cuda"

REM Activate Conda environment
CALL conda activate %CONDA_ENV%

REM Run the command with the defined variables
python train_expert_transfer_modified_dynamics.py ^
    --reward_model "%REWARD_MODEL%" ^
    --modified_xml "%MODIFIED_XML%" ^
    %CUDA% ^
    --env_id "%ENV_ID%" ^
    --num_steps "%NUM_STEPS%" ^
    --eval_interval "%EVAL_INTERVAL%" ^
    --seed "%SEED%"

REM Deactivate Conda environment after execution
CALL conda deactivate
