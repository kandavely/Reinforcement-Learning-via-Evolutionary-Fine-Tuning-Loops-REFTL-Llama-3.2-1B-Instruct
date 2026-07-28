@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Configuration ---
SET SCRIPT_GEN="Question and Answer Generator 2.py"
SET SCRIPT_FT="fine tuning base model 2.py"
SET SCRIPT_EVAL="evaluation_script.py"
SET TOTAL_LOOPS=10
SET EVOLUTIONARY_NODE_NUMBERS=3

SET RESTART_COUNT=0
SET RESTART_LIMIT=10


REM ---

ECHO.
ECHO =========================================================
ECHO Starting Continuous Fine-Tuning Loop
ECHO Total Outer Iterations: %TOTAL_LOOPS%
ECHO Evolutionary Nodes per Iteration: %EVOLUTIONARY_NODE_NUMBERS%
ECHO =========================================================
ECHO.

REM The outer loop runs for TOTAL_LOOPS iterations 
FOR /L %%i IN (1, 1, %TOTAL_LOOPS%) DO (

    ECHO #########################################################
    ECHO # STARTING OUTER ITERATION %%i of %TOTAL_LOOPS%
    ECHO #########################################################
    ECHO.

    REM --- INNER LOOP: Evolutionary Nodes ---
    ECHO Starting Inner Evolutionary Loop...
    FOR /L %%j IN (1, 1, %EVOLUTIONARY_NODE_NUMBERS%) DO (
        
        ECHO.
        ECHO   [Node %%j of %EVOLUTIONARY_NODE_NUMBERS%]

        :START_DATA_GENERATION
        REM --- Step 1: Data Generation (NOW INSIDE INNER LOOP) ---

        ECHO   --- 1. Executing Data Generation Script: !SCRIPT_GEN! ---
        python !SCRIPT_GEN!
        
        IF ERRORLEVEL 1 ( 
            

            ECHO   !!! ERROR: Data Generation failed at Node %%j. !!!
            SET /A RESTART_COUNT+=1

            REM Sleep 60 seconds before retrying
            TIMEOUT /T 60 /NOBREAK

            IF !RESTART_COUNT! GEQ %RESTART_LIMIT% (
                ECHO   !!! Restart limit reached. Exiting script. !!!
                GOTO :END
            )


            GOTO :START_DATA_GENERATION
        )


        ECHO   Data Generation Complete for Node %%j. 
        
        :START_FINETUNING
        REM --- Step 2: Fine-Tuning ---
        ECHO   --- 2. Executing Fine-Tuning Script: !SCRIPT_FT! --- 
        python !SCRIPT_FT!
      
        IF ERRORLEVEL 1 ( 
            ECHO   !!! ERROR: Fine-Tuning failed at Node %%j. !!!
            
            SET /A RESTART_COUNT+=1

            REM Sleep 60 seconds before retrying
            TIMEOUT /T 60 /NOBREAK


            IF !RESTART_COUNT! GEQ %RESTART_LIMIT% (
                ECHO   !!! Restart limit reached. Exiting script. !!!
                GOTO :END
            )



            GOTO :START_FINETUNING
        )
        ECHO   Fine-Tuning Complete for Node %%j. 
        
        :START_EVALUATION
        REM --- Step 3: Evaluation ---
        ECHO   --- 3. Executing Evaluation Script: !SCRIPT_EVAL! --- 
        python !SCRIPT_EVAL!
        


        IF ERRORLEVEL 1 ( 
            ECHO   !!! ERROR: Evaluation failed at Node %%j. !!!

            SET /A RESTART_COUNT+=1

            REM Sleep 60 seconds before retrying
            TIMEOUT /T 60 /NOBREAK


            IF !RESTART_COUNT! GEQ %RESTART_LIMIT% (
                ECHO   !!! Restart limit reached. Exiting script. !!!
                GOTO :END
            )



            GOTO :START_EVALUATION
        )
        ECHO   Evaluation Complete for Node %%j.
    )
    REM --- END OF INNER LOOP ---

    ECHO.
    ECHO =========================================================
    ECHO OUTER ITERATION %%i COMPLETE.
    ECHO =========================================================
    ECHO.
)

:END
ECHO.
ECHO =========================================================
ECHO Reinforcement Learning using Evolutionary Fine-Tuning Loop Finished.
ECHO =========================================================
ECHO.
PAUSE