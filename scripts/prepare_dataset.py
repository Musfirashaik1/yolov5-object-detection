"""
Split a YOLO-format dataset into train/val/test and generate data.yaml.

Expected input layout:
    raw_data/images/*.jpg|png
    raw_data/labels/*.txt      (YOLO format: class x_center y_center width height, normalized)
    raw_data/classes.txt       (one class name per line)

Usage:
    python scripts/prepare_dataset.py --src raw_data --dst dataset
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_data")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {src / 'images'}")

    random.Random(args.seed).shuffle(images)
    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        missing = 0
        for img in files:
            shutil.copy2(img, dst / "images" / split / img.name)
            label = src / "labels" / (img.stem + ".txt")
            if label.exists():
                shutil.copy2(label, dst / "labels" / split / label.name)
            else:
                missing += 1  # background image (no objects)
        print(f"{split}: {len(files)} images ({missing} without labels)")

    classes = [c.strip() for c in (src / "classes.txt").read_text().splitlines() if c.strip()]
    yaml_text = f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: {len(classes)}\nnames:\n"
    yaml_text += "".join(f"  {i}: {name}\n" for i, name in enumerate(classes))
    (dst / "data.yaml").write_text(yaml_text)
    print(f"Wrote {dst / 'data.yaml'} with {len(classes)} classes")


if __name__ == "__main__":
    main()
