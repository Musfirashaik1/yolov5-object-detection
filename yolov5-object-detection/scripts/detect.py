"""
Run object detection with YOLOv5 on an image, a folder of images, a video, or a webcam.

Examples:
    python scripts/detect.py --source photo.jpg
    python scripts/detect.py --source video.mp4 --weights runs/train/custom_run/weights/best.pt
    python scripts/detect.py --source 0                      # webcam (press 'q' to quit)
    python scripts/detect.py --source images/ --conf 0.4 --no-show
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(weights, conf, iou, device):
    if Path(weights).exists():  # custom-trained weights
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
    else:                       # pretrained COCO model name, e.g. yolov5s
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)
    model.conf = conf
    model.iou = iou
    model.to(device)
    return model


def color_for(cls_id):
    rng = np.random.RandomState(cls_id * 7 + 3)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw(frame, detections):
    for _, r in detections.iterrows():
        x1, y1, x2, y2 = int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)
        color = color_for(int(r["class"]))
        label = f"{r['name']} {r.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def detect_frame(model, frame_bgr, size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=size)
    return results.pandas().xyxy[0]


def process_images(model, paths, out_dir, size, show):
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"Could not read {p}")
            continue
        det = detect_frame(model, frame, size)
        out = draw(frame, det)
        cv2.imwrite(str(out_dir / p.name), out)
        print(f"{p.name}: {len(det)} objects -> {det['name'].value_counts().to_dict()}")
        if show:
            cv2.imshow("YOLOv5", out)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break


def process_stream(model, cap, out_path, size, show):
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        det = detect_frame(model, frame, size)
        frame = draw(frame, det)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}  Objects: {len(det)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)
        if show:
            cv2.imshow("YOLOv5", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    writer.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image / folder / video path, or webcam index like 0")
    ap.add_argument("--weights", default="yolov5s", help="path to best.pt, or pretrained name (yolov5s/m/l/x)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--out", default="runs/detect")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, args.conf, args.iou, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    if args.source.isdigit():
        process_stream(model, cv2.VideoCapture(int(args.source)), out_dir / "webcam.mp4", args.img, show)
    else:
        src = Path(args.source)
        if src.is_dir():
            process_images(model, sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS), out_dir, args.img, show)
        elif src.suffix.lower() in IMG_EXTS:
            process_images(model, [src], out_dir, args.img, show)
        elif src.suffix.lower() in VID_EXTS:
            process_stream(model, cv2.VideoCapture(str(src)), out_dir / (src.stem + "_det.mp4"), args.img, show)
        else:
            raise SystemExit(f"Unsupported source: {src}")

    cv2.destroyAllWindows()
    print(f"Results saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
