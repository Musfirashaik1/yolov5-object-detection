"""
Evaluate trained weights on the validation or test split (precision, recall, mAP).

Usage:
    python scripts/evaluate.py --weights runs/train/custom_run/weights/best.pt --task test
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--task", default="val", choices=["val", "test"])
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    cmd = [
        sys.executable, str(YOLO_DIR / "val.py"),
        "--weights", args.weights,
        "--data", str(Path(args.data).resolve()),
        "--img", str(args.img),
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--task", args.task,
        "--project", str(ROOT / "runs" / "val"),
        "--name", "eval",
        "--exist-ok",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
