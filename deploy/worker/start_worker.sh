#!/bin/bash
# Vast SSH mode replaces the image command and then runs this script.
# setsid moves the server into its own session so it stays up after this
# shell exits. The log records the launch and contains no environment values.
set -u
log="${WORKER_LOG:-/workspace/worker-ready.log}"
pidfile="${WORKER_PID_FILE:-/workspace/worker-ready.pid}"
script="${WORKER_SCRIPT:-/workspace/worker_ready.py}"
python="${WORKER_PYTHON:-python3}"
mkdir -p "$(dirname "$log")" "$(dirname "$pidfile")"
stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if "$python" -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/health" % os.environ.get("WORKER_PORT", "8080"), timeout=2)' >/dev/null 2>&1; then
  printf '%s already_listening\n' "$stamp" >> "$log"
  exit 0
fi
printf '%s launching\n' "$stamp" >> "$log"
setsid "$python" "$script" >> "$log" 2>&1 < /dev/null &
echo $! > "$pidfile"
exit 0
