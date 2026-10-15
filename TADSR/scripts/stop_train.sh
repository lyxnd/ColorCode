#!/bin/bash
# [LLIE] 停止 scripts/train_bg.sh 启动的后台训练（连同 accelerate 启动的所有子进程一起停掉）
cd "$(dirname "$0")/.." || exit 1
if [ ! -f logs/train.pid ]; then
    echo "没有找到 logs/train.pid，可能没有通过 train_bg.sh 启动过训练"; exit 1
fi
PID=$(cat logs/train.pid)
if ! kill -0 "$PID" 2>/dev/null; then
    echo "进程 $PID 已经不在运行"; rm -f logs/train.pid; exit 0
fi
# train_bg.sh 用 setsid 启动，进程组号 = PID，向整个进程组发信号即可停掉所有子进程
kill -TERM -- "-$PID"
for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
    echo "30 秒内没有退出，强制结束"
    kill -KILL -- "-$PID"
fi
rm -f logs/train.pid
echo "训练已停止 (PID $PID)"
