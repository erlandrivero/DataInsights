@echo off
git add utils/ml_training.py utils/ml_regression.py
git commit -m "Fix datetime column handling in ML Classification and Regression"
git push origin main
pause
