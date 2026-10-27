@echo off
title X-FLIP 14 MASTER ORCHESTRATOR (ADMIN)
cd /d C:\cluster

echo ==========================================================
echo   VERIFYING 3-NODE CLUSTER CONNECTIVITY (192.168.150.x)
echo ==========================================================
ping -n 1 192.168.150.216 >nul && (echo [OK] Device 1: HP 840 G7 ^(192.168.150.216^) is ONLINE) || (echo [WARN] HP 840 G7 unreachable!)
ping -n 1 192.168.150.166 >nul && (echo [OK] Device 3: Dell Kali ^(192.168.150.166^) is ONLINE) || (echo [WARN] Dell Kali unreachable!)

echo.
echo Launching Lead OpenCode Orchestrator in Auto Mode...
opencode --auto
