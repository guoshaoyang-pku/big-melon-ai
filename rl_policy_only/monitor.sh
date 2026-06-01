#!/bin/zsh
# Suika AlphaZero actor-learner training monitor / control helper.
cd "$(dirname "$0")"
PIDFILE=logs/train.pid
LEARNER=$(cat "$PIDFILE" 2>/dev/null)
# comma-joined child pids (macOS `ps -p` wants a comma list)
wcsv() { [ -n "$LEARNER" ] && pgrep -P "$LEARNER" 2>/dev/null | paste -sd, - ; }
allcsv() { local w=$(wcsv); echo "${LEARNER}${w:+,$w}"; }
case "$1" in
  log)    tail -f logs/train.log ;;
  status) echo "learner PID: $LEARNER"; \
          ps -p "$LEARNER" -o pid,stat,%cpu,rss,etime 2>/dev/null; \
          echo "--- recent STATS / EVAL ---"; \
          grep -E "STATS|EVAL|RESUME|shutdown|snapshot" logs/train.log | tail -8 ;;
  ckpt)   ls -la checkpoints ;;
  cpu)    echo "== load avg (10 cores) =="; uptime; \
          echo "== learner + workers (%CPU / RSS-KB) =="; \
          ps -p "$(allcsv)" -o pid,%cpu,rss,etime,comm 2>/dev/null ;;
  mem)    ps -p "$(allcsv)" -o rss= 2>/dev/null | \
          awk '{s+=$1} END {printf "total RSS: %.2f GB across %d procs\n", s/1024/1024, NR}' ;;
  busy)   echo "== per-worker %CPU (all should be busy ~80-100%) =="; \
          ps -p "$(wcsv)" -o pid,%cpu,etime 2>/dev/null; \
          echo "== games per worker (from last STATS) =="; \
          grep "STATS" logs/train.log | tail -1 | grep -oE "workers\[.*\]" ;;
  stop)   echo "sending TERM to $LEARNER (graceful: drains queue, saves ckpt+buffer)"; \
          kill "$LEARNER" 2>/dev/null && echo "sent" || echo "no process" ;;
  *) echo "usage: ./monitor.sh {log|status|ckpt|cpu|mem|busy|stop}" ;;
esac
