@echo off
REM Batch script to generate demonstrations for multiple environments
REM Make sure Conda is initialized for cmd.exe (run 'conda init cmd.exe' once if needed)

ECHO Starting demo generation for InvertedPendulum-v5...
conda run -n honors-capstone python scripts/generate_demos.py --env InvertedPendulum-v5 --weights ./logs/InvertedPendulum-v5/sac/step100000 --std 0.01 --cuda
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error generating demos for InvertedPendulum-v5. Exiting.
    GOTO :EOF
)
ECHO Finished InvertedPendulum-v5.
ECHO.

ECHO Starting demo generation for Hopper-v5...
conda run -n honors-capstone python scripts/generate_demos.py --env Hopper-v5 --weights ./logs/Hopper-v5/sac/step1000000 --std 0.01 --cuda
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error generating demos for Hopper-v5. Exiting.
    GOTO :EOF
)
ECHO Finished Hopper-v5.
ECHO.

ECHO Starting demo generation for Ant-v5...
conda run -n honors-capstone python scripts/generate_demos.py --env Ant-v5 --weights ./logs/Ant-v5/sac/step1000000 --std 0.01 --cuda
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error generating demos for Ant-v5. Exiting.
    GOTO :EOF
)
ECHO Finished Ant-v5.
ECHO.

ECHO Starting demo generation for Pusher-v5...
conda run -n honors-capstone python scripts/generate_demos.py --env Pusher-v5 --weights ./logs/Pusher-v5/sac/step1000000 --std 0.01 --cuda
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error generating demos for Pusher-v5. Exiting.
    GOTO :EOF
)
ECHO Finished Pusher-v5.
ECHO.

ECHO All demo generations completed successfully.

:EOF
