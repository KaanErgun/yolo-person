#!/bin/bash

# Training başlatma scripti
# SSH bağlantısı koparsa bile devam eder

SESSION_NAME="yolo_training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Mevcut session varsa attach et
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Training session zaten çalışıyor. Bağlanmak için:"
    echo "tmux attach -t $SESSION_NAME"
    exit 0
fi

# Yeni tmux session oluştur
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

# Virtual environment'ı aktifleştir ve training'i başlat
tmux send-keys -t $SESSION_NAME "source venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python train.py 2>&1 | tee -a training.log" C-m

echo "Training başladı! Detayları görmek için:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Session'dan ayrılmak için (training devam eder): Ctrl+B sonra D"
echo ""
echo "Training durumunu kontrol etmek için:"
echo "  tail -f training.log"
