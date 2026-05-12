# Real-Time Object Detection with YOLOv8

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## About

Real-time object detection system using **YOLOv8** and **OpenCV**. Detects and identifies objects from your webcam in real-time with bounding boxes and confidence scores.

**Key Features:**
- Real-time detection at 30+ FPS
- Webcam stream processing
- Accurate bounding boxes with confidence scores
- Class labels for each detected object
- Color-coded boxes and text overlay
- Easy to customize and extend

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Language** | Python 3.8+ |
| **Pre-trained Model** | COCO Dataset (80 classes) |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (or any video input device)

### Step 1: Clone the Repository
```bash
git clone https://github.com/FibyEhab/real-time-object-detection.git
cd real-time-object-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Or Install Manually
```bash
pip install ultralytics opencv-python
```

---

## Usage

### Run the Detection Script
```bash
python main.py
```

### What Happens
1. Opens your webcam
2. Processes frames in real-time using YOLOv8
3. Draws bounding boxes around detected objects
4. Shows class label and confidence score
5. Press **Q** to quit

### Code Breakdown

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    results = model.predict(frame, stream=True, verbose=False, conf=0.5)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Real Time Detection", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Performance

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8s (Small) |
| **FPS** | 30+ on CPU / 100+ on GPU |
| **Input Resolution** | 1920×1080 |
| **Confidence Threshold** | 0.5 (adjustable) |
| **Classes Detected** | 80 (COCO Dataset) |

---

## Customization

### Use Different YOLOv8 Versions
```python
# Nano (fastest, less accurate)
model = YOLO("yolov8n.pt")

# Small (balanced)
model = YOLO("yolov8s.pt")

# Medium (more accurate)
model = YOLO("yolov8m.pt")

# Large (most accurate, slowest)
model = YOLO("yolov8l.pt")

# Extra Large
model = YOLO("yolov8x.pt")
```

### Adjust Confidence Threshold
```python
# Only detect objects with >70% confidence
results = model.predict(frame, conf=0.7)
```

### Change Video Source
```python
# Use video file instead of webcam
cap = cv2.VideoCapture("path/to/video.mp4")

# Use IP camera
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
```

### Change Colors and Styles
```python
# Change bounding box color (BGR format)
cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue

# Change text color
cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Cyan
```

---

## Project Structure

```
real-time-object-detection/
├── main.py                 # Main detection script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── images/
│   └── demo.png           # Demo screenshot
└── LICENSE                # MIT License
```

---

## Detectable Classes

The YOLOv8 model trained on COCO dataset can detect 80 classes including:

**People & Animals:**
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe

**Indoor Objects:**
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Outdoor Objects:**
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

[See full COCO dataset](https://cocodataset.org/#home)

---

## Performance Tips

### For Better Speed:
- Use **YOLOv8n** (nano) instead of larger models
- Lower input resolution
- Increase confidence threshold

### For Better Accuracy:
- Use **YOLOv8l** or **YOLOv8x** models
- Increase input resolution
- Lower confidence threshold

### GPU Acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### **Issue: Webcam not working**
```bash
# Check if webcam is recognized
python -c "import cv2; print(cv2.VideoCapture(0).get(cv2.CAP_PROP_FRAME_WIDTH))"
```

### **Issue: Slow detection**
- Use a smaller model (YOLOv8n)
- Use GPU acceleration
- Lower input resolution

### **Issue: Objects not detected**
- Lower the confidence threshold
- Check lighting conditions
- Try a larger model

### **Issue: ModuleNotFoundError**
```bash
pip install --upgrade ultralytics opencv-python
```

---

## Results

- Real-time detection on CPU
- High accuracy with YOLOv8s
- Works with any webcam or video input
- Easily customizable and extendable

**Example Detections:**
- Person with 95% confidence
- Car with 87% confidence
- Dog with 92% confidence
- Chair with 78% confidence

---

## Learning Resources

- [YOLOv8 Official Docs](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLO Paper](https://arxiv.org/abs/2004.10934)
- [COCO Dataset](https://cocodataset.org/)

---

## Contributing

Got ideas for improvements? I'd love to collaborate!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**YOLOv8 Model:** Licensed under the AGPL-3.0 License by Ultralytics

---

## Author

**Fiby Ehab** - AI Engineer

**Email:** febeehab3@gmail.com

**LinkedIn:** [Fiby Ehab](https://www.linkedin.com/in/fiby-ehab-270b55286/)

**GitHub:** [@FibyEhab](https://github.com/FibyEhab)  

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision tools
- [COCO Dataset](https://cocodataset.org/) for pre-trained weights

---

## ⭐ If this project helped you, please star it!

Made with ❤️ for the AI community | Last updated: January 2025
