@echo off
echo ====================================
echo DataInsights - Phase 1 Deployment
echo ====================================
echo.

echo Step 1: Committing changes...
git commit -m "Phase 1: Add error handling, validation, rate limiting, tests, and fix branding"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git commit failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Pushing to GitHub...
git push origin main

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git push failed!
    pause
    exit /b 1
)

echo.
echo ====================================
echo SUCCESS! Deployment complete!
echo ====================================
echo.
echo Your changes are now deploying to Streamlit Cloud.
echo Visit your Streamlit Cloud dashboard to monitor deployment.
echo.
pause
