"""
Train YOLOv5 on your custom dataset (wrapper around the official train.py).

Usage:
    python scripts/train.py --data dataset/data.yaml --epochs 100 --model yolov5s
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolov5s", help="yolov5n/s/m/l/x (pretrained weights are downloaded)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    ap.add_argument("--name", default="custom_run")
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience")
    args = ap.parse_args()

    if not YOLO_DIR.exists():
        raise SystemExit("yolov5/ folder not found. Run ./setup.sh first.")

    cmd = [
        sys.executable, str(YOLO_DIR / "train.py"),
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", str(Path(args.data).resolve()),
        "--weights", f"{args.model}.pt",
        "--project", str(ROOT / "runs" / "train"),
        "--name", args.name,
        "--patience", str(args.patience),
        "--cache",
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. Best weights: runs/train/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
