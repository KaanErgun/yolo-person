#!/bin/bash

# Training monitoring scripti
# GPU ve training durumunu takip eder

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/training.log"
MONITOR_LOG="$SCRIPT_DIR/monitor.log"

echo "=== Training Monitor ===" | tee -a "$MONITOR_LOG"
echo "Başlangıç: $(date)" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

# GPU durumu
echo "GPU Durumu:" | tee -a "$MONITOR_LOG"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

# Training log'dan son durumu al
if [ -f "$LOG_FILE" ]; then
    echo "Training Durumu:" | tee -a "$MONITOR_LOG"
    
    # Son epoch bilgisi ve progress hesaplama - daha hızlı versiyon
    LAST_EPOCH=$(tail -200 "$LOG_FILE" | grep -oE "[0-9]+/100" | tail -1 | cut -d'/' -f1)
    if [ -n "$LAST_EPOCH" ]; then
        PROGRESS=$((LAST_EPOCH * 100 / 100))
        REMAINING=$((100 - LAST_EPOCH))
        
        # Her epoch ~30 dakika sürdüğünü varsayalım (batch=4, workers=2 ile)
        MINUTES_PER_EPOCH=30
        REMAINING_MINUTES=$((REMAINING * MINUTES_PER_EPOCH))
        REMAINING_HOURS=$((REMAINING_MINUTES / 60))
        REMAINING_MINS=$((REMAINING_MINUTES % 60))
        
        echo "  📊 Epoch: $LAST_EPOCH/100 (%${PROGRESS} tamamlandı)" | tee -a "$MONITOR_LOG"
        echo "  ⏰ Kalan: $REMAINING epoch (~${REMAINING_HOURS}h ${REMAINING_MINS}m)" | tee -a "$MONITOR_LOG"
        
        # Tahmini bitiş zamanı
        END_TIME=$(date -d "+${REMAINING_MINUTES} minutes" "+%Y-%m-%d %H:%M:%S")
        echo "  🏁 Tahmini Bitiş: $END_TIME" | tee -a "$MONITOR_LOG"
        echo "" | tee -a "$MONITOR_LOG"
    fi
    
    # Son loss değerleri - daha temiz gösterim
    echo "  📉 Son Loss Değerleri:" | tee -a "$MONITOR_LOG"
    tail -50 "$LOG_FILE" | grep "box_loss.*cls_loss.*dfl_loss" | tail -1 | tee -a "$MONITOR_LOG"
    
    # Validation sonuçları varsa - sadece özet
    echo "" | tee -a "$MONITOR_LOG"
    echo "  📈 Son Validation Sonuçları:" | tee -a "$MONITOR_LOG"
    tail -50 "$LOG_FILE" | grep "all.*mAP" | tail -1 | tee -a "$MONITOR_LOG"
    
    echo "" | tee -a "$MONITOR_LOG"
    
    # Training devam ediyor mu kontrolü
    if pgrep -f "python train.py" > /dev/null; then
        echo "✅ Training aktif olarak devam ediyor" | tee -a "$MONITOR_LOG"
    else
        echo "⚠️  Training process bulunamadı" | tee -a "$MONITOR_LOG"
    fi
else
    echo "Training log dosyası henüz oluşturulmamış" | tee -a "$MONITOR_LOG"
fi

echo "" | tee -a "$MONITOR_LOG"
echo "======================================" | tee -a "$MONITOR_LOG"
