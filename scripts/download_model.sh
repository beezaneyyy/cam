#!/usr/bin/env bash
# Downloads the MediaPipe hand landmarker model (≈7.5 MB, one time).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
echo "[airframe] downloading hand model…"
curl -fL --progress-bar -o models/hand_landmarker.task.part "$URL"
mv models/hand_landmarker.task.part models/hand_landmarker.task
echo "[airframe] model saved to models/hand_landmarker.task"
