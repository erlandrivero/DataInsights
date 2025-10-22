@echo off
git add app.py
git commit -m "Fix: Block ML training when classes will have <2 samples after sampling"
git push origin main
pause
