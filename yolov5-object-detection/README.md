# Object Detection using YOLOv5

End-to-end object detection project: dataset preparation, training, evaluation,
real-time inference (image / video / webcam) and model export, built on
PyTorch, OpenCV and the official Ultralytics YOLOv5 repository.

## Demo

   ![YOLOv5 detection demo](assets/bus_detection.jpg)

   Pretrained YOLOv5s detecting a bus and people (confidence score shown next to each label).

## Project structure

```
yolov5-object-detection/
├── setup.sh                    # clones YOLOv5 + installs dependencies
├── requirements.txt
├── raw_data/
│   ├── images/                 # put your images here
│   ├── labels/                 # YOLO-format .txt labels (same file names)
│   └── classes.txt             # one class name per line
├── scripts/
│   ├── prepare_dataset.py      # train/val/test split + data.yaml
│   ├── train.py                # training wrapper
│   ├── evaluate.py             # precision / recall / mAP
│   ├── detect.py               # image / folder / video / webcam inference
│   └── export_model.py         # ONNX / TorchScript export
└── runs/                       # outputs (created automatically)
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
./setup.sh                                          # Windows: run the git clone + pip commands manually
```

## 2. Try it instantly with pretrained COCO weights (no training needed)

```bash
python scripts/detect.py --source 0                       # webcam, press q to quit
python scripts/detect.py --source path/to/image.jpg
python scripts/detect.py --source path/to/video.mp4 --no-show
```

## 3. Train on your own dataset

1. Label images in YOLO format using LabelImg, CVAT, Roboflow or Label Studio.
   Each label line is: `class_id x_center y_center width height` (all normalized 0-1).
2. Place images in `raw_data/images/`, labels in `raw_data/labels/`, and list your classes in `raw_data/classes.txt`.
3. Split the data and generate `data.yaml`:
   ```bash
   python scripts/prepare_dataset.py --src raw_data --dst dataset
   ```
4. Train:
   ```bash
   python scripts/train.py --epochs 100 --model yolov5s --batch 16
   ```
   Best weights are saved to `runs/train/custom_run/weights/best.pt`.

## 4. Evaluate

```bash
python scripts/evaluate.py --weights runs/train/custom_run/weights/best.pt --task test
```

Reports precision, recall, mAP@0.5 and mAP@0.5:0.95 per class and overall.

## 5. Detect with your trained model

```bash
python scripts/detect.py --source test.jpg --weights runs/train/custom_run/weights/best.pt --conf 0.4
```

## 6. Export for deployment

```bash
python scripts/export_model.py --weights runs/train/custom_run/weights/best.pt --include onnx
```

## Tips for better accuracy

- Aim for 1,500+ images per class with varied lighting, angles and backgrounds.
- Include some background images (no objects) to reduce false positives.
- Start with `yolov5s`; move to `yolov5m` or `yolov5l` if accuracy is low and you have a GPU.
- Check `runs/train/<name>/results.png`: if validation loss rises while training loss falls, you are overfitting.
- Use a free GPU on Google Colab or Kaggle Notebooks if you have none locally.

## Tech stack

Python, PyTorch, OpenCV, YOLOv5 (Ultralytics), NumPy, Pandas
