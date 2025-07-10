# YOLO Vehicle In/Out Tracking 🚗🔄

This project uses **YOLOv8** (You Only Look Once) for **real-time vehicle detection and tracking**, and logs vehicles entering or exiting a predefined zone using a virtual line or ROI (Region of Interest).

## 📦 Features

- Vehicle Detection using YOLOv8
- Object Tracking (DeepSORT or ByteTrack)
- Entry/Exit Zone line counting
- Logs in/out timestamps
- RTSP/Live camera/video file support

## 🔧 Requirements

```bash
pip install ultralytics opencv-python numpy
# Optional for tracking
pip install deep-sort-realtime