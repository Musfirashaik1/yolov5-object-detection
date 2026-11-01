#!/usr/bin/env bash
# One-time setup: clone the official YOLOv5 repo and install dependencies.
set -e
if [ ! -d "yolov5" ]; then
  git clone https://github.com/ultralytics/yolov5.git
fi
pip install -r requirements.txt
pip install -r yolov5/requirements.txt
echo "Setup complete."
