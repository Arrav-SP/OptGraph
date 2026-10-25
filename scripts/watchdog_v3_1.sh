#!/usr/bin/env bash
# Keeps the OptGraph v3.1 full run going across WSL/Windows restarts. Run every 15 minutes by the
# Windows scheduled task "OptGraph v3.1 watchdog" (which also starts WSL if it is down).
# Does nothing when the run is finished (data/v3_1/RUN_COMPLETE) or still alive (PID file + args).
# Every stage of the chain is resumable, so a restart continues where the run stopped.
set -u
cd "$(dirname "$0")/.."
W=~/optgraph_work
LOG=$W/v3_1_watchdog.log
ts() { TZ=Asia/Kolkata date +"%Y-%m-%d %H:%M:%S IST"; }
[ -f data/v3_1/RUN_COMPLETE ] && exit 0
# A Windows-side client holding WSL open, so WSL does not stop when no editor is attached.
if ! ps -eo args | awk '$1 == "optgraph-keepalive" {f=1} END {exit !f}'; then
  /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
    "Start-Process -WindowStyle Hidden conhost.exe -ArgumentList '--headless','wsl.exe','-d','Ubuntu','-u','arav','--','bash','$PWD/scripts/keepalive_v3_1.sh'" >/dev/null 2>&1
  echo "$(ts) keep-alive started" >> $LOG
fi
pid=$(cat $W/v3_1_full.pid 2>/dev/null || echo 0)
if [ "$pid" -gt 0 ] && tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "run_v3_1.sh"; then
  exit 0
fi
echo "$(ts) run not alive (pid $pid); uptime: $(uptime -p); restarting" >> $LOG
[ -f $W/v3_1_full.log ] && cp $W/v3_1_full.log "$W/v3_1_full_$(date +%s).log"
source ~/.venvs/optgraph/bin/activate
setsid nohup bash -c 'export OPTGRAPH_CONFIG=v3_1.yaml; bash scripts/run_v3_1.sh && bash scripts/run_v3_1.sh --runtime && python scripts/v3_1_report.py && touch data/v3_1/RUN_COMPLETE' \
  > $W/v3_1_full.log 2>&1 < /dev/null &
echo $! > $W/v3_1_full.pid
echo "$(ts) started chain pid $!" >> $LOG
