@echo off
echo ============================================
echo   DeepTruth Startup Verification
echo ============================================

:: Check model file exists
if not exist "models\deeptruth_image_model_final.pth" (
    echo [ERROR] Image model NOT FOUND: models\deeptruth_image_model_final.pth
    echo         Restore from backup or retrain.
    pause
    exit /b 1
)

:: Check model file size (should be ~519 MB = 507067 KB)
for %%A in ("models\deeptruth_image_model_final.pth") do set SIZE=%%~zA
echo [INFO] Image model size: %SIZE% bytes

if %SIZE% LSS 500000000 (
    echo [WARNING] Model file is smaller than expected ^(~519 MB^).
    echo           It may be a wrong or incomplete model.
    pause
)

:: Check MD5 of image model
echo [INFO] Checking model integrity...
for /f "skip=1 tokens=*" %%A in ('certutil -hashfile "models\deeptruth_image_model_final.pth" MD5') do (
    if not defined MD5HASH set MD5HASH=%%A
)
echo [INFO] Current MD5:  %MD5HASH%
echo [INFO] Expected MD5: 1f07bc973f7079374f91c8ca84a77f42

if /i "%MD5HASH%"=="1f07bc973f7079374f91c8ca84a77f42" (
    echo [OK] Model file integrity verified.
) else (
    echo [WARNING] Model MD5 does not match the known-good version.
    echo           The model may have been replaced. Detection results may be wrong.
    choice /C YN /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

echo.
echo ============================================
echo   Starting Backend
echo ============================================
cd backend
start "DeepTruth Backend" cmd /k "..\\.venv\\Scripts\\uvicorn app.main:app --host 0.0.0.0 --port 8000"
cd ..

timeout /t 5 /nobreak >nul

echo ============================================
echo   Starting Frontend
echo ============================================
cd frontend
start "DeepTruth Frontend" cmd /k "npm run dev"
cd ..

echo.
echo Both servers started.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
