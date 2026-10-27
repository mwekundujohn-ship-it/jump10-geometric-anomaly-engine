@echo off
cd /d C:\cluster\phase1
python C:\cluster\phase1\phase1_worker.py --job %1 > %2 2>&1
