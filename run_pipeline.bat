@echo off
REM =====================================================================
REM Facial Age Classification Pipeline & Report Generator Runner
REM Student: Rahim Ullah (FA24-BSE-051)
REM =====================================================================

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo =====================================================================
echo  FACIAL AGE CLASSIFICATION ^& DEEP LEARNING BENCHMARK
echo  Ordinal Ridge Regression vs. Random Forest vs. Deep CNN
echo =====================================================================
echo [INFO] Environment: .\.venv\Scripts\python.exe
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\.venv\
    echo Please make sure the .venv directory is present.
    pause
    exit /b 1
)

echo Running 3-Way Pipeline (PCA + Ridge + RF + Deep CNN)...
".\.venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed during execution.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo  ALL TASKS COMPLETED SUCCESSFULLY!
echo =====================================================================
echo  * Generated Graphs (8 visual figures) in processed\
echo  * Standalone Notebook: Facial_Age_ML_Classification_and_Regression_Complete_Assignment.ipynb
echo =====================================================================
pause
