#!/usr/bin/env bash
# Learned-aggregator arm (AGG=attn|gru|swin, default swin): same pipeline as the mean control.
L=/scratch/bambi/alfs_embed/logs/viewagg
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -n 2
cd /scratch/bambi/alfs_embed/code
for m in thermal:6 rgb:7; do
  mod=${m%%:*}; gpu=${m##*:}
  AGG=${AGG:-swin} HID=${HID:-64} WIN=${WIN:-4} setsid nohup bash docker/viewagg_cv.sh "$mod" "$gpu" > "$L/nohup_${AGG:-swin}_$mod.out" 2>&1 < /dev/null &
done
sleep 5
pgrep -af viewagg_cv | grep -v pgrep
tail -n 2 $L/viewagg_thermal.log $L/viewagg_rgb.log
