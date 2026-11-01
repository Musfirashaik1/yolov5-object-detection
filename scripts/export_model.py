"""
Export trained weights to ONNX (or other formats) for deployment.

Usage:
    python scripts/export_model.py --weights runs/train/custom_run/weights/best.pt --include onnx
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--include", nargs="+", default=["onnx"], help="onnx torchscript tflite engine ...")
    ap.add_argument("--img", type=int, default=640)
    args = ap.parse_args()

    cmd = [sys.executable, str(ROOT / "yolov5" / "export.py"),
           "--weights", args.weights, "--img", str(args.img), "--include", *args.include]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
