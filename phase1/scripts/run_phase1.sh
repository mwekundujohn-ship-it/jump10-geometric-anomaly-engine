#!/bin/bash
cd /home/user/cluster/phase1 || exit 1
python3 phase1_worker.py --job "$1" > "$2" 2>&1
