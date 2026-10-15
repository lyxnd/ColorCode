#!/bin/bash
# [LLIE] 后台启动训练：SSH 断开后训练继续运行，输出写入 logs/ 下的日志文件。
# 用法（在项目根目录或任意位置，先激活好训练用的 conda 环境）：
#   bash scripts/train_bg.sh                         # 默认运行 scripts/train_tadsr.sh
#   bash scripts/train_bg.sh scripts/其他训练脚本.sh   # 运行指定脚本
# 查看日志：tail -f logs/train_latest.log
# 停止训练：bash scripts/stop_train.sh

cd "$(dirname "$0")/.." || exit 1
TRAIN_SCRIPT="${1:-scripts/train_tadsr.sh}"
mkdir -p logs

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "找不到训练脚本: $TRAIN_SCRIPT"; exit 1
fi

# 已有训练在运行时不重复启动
if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
    echo "已有训练在运行 (PID $(cat logs/train.pid))，如需重启先执行: bash scripts/stop_train.sh"; exit 1
fi

LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1   # 让 print 立即写入日志，而不是攒在缓冲区里

# setsid：新建会话，脱离当前终端，SSH 断开时不会收到 SIGHUP；nohup 作为双保险
setsid nohup bash "$TRAIN_SCRIPT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > logs/train.pid
ln -sf "$(basename "$LOG")" logs/train_latest.log

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "训练已在后台启动 (PID $PID)"
    echo "日志:     $LOG"
    echo "查看日志: tail -f logs/train_latest.log"
    echo "停止训练: bash scripts/stop_train.sh"
else
    echo "训练进程启动后立即退出了，看日志排查: $LOG"
    tail -n 20 "$LOG"
    exit 1
fi
