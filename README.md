# ADAS: Advanced Driver Assistance System Simulation

This repository contains the code and report for the **Advanced Driver Assistance System (ADAS)** project, a virtual driver aid developed and tested within the **Grand Theft Auto V (GTA-5)** environment. The system employs real-time object detection, lane detection, and automated driving decisions using computer vision and machine learning techniques.

---

## 🚗 Project Overview

The ADAS project simulates autonomous driving assistance capabilities using the GTA-5 environment. It aims to:
- Detect objects (vehicles, pedestrians)
- Recognize lane markings
- Predict and perform actions like steering and braking
- Steer a virtual car in real-time

---

## 🧠 Features & Algorithms

### Object Detection
- **YOLOv8** model for real-time detection of cars, trucks, pedestrians.
- Distance estimation based on bounding box size and camera parameters.

### Action Prediction
- **Support Vector Machine (SVM)** predicts actions (`left`, `right`, `brake`, `forward`) using positional features.
- Trained with a custom dataset of labeled driving actions.

### Lane Detection
- **Canny Edge Detection** for edge filtering.
- **Region of Interest Masking** to focus on relevant areas.
- **Hough Line Transform** to detect lane markings.
- **Heuristic-based Auto Steering** based on lane line orientation.
- **Real-time Continuous Loop** using image capture for dynamic lane updates.

### Destination Reaching Strategy
- Detects in-game purple paths from the mini-map to reach destinations.

---

## 📊 Datasets & Training

- **Custom dataset** with features: `Avg_X`, `Avg_Y`, `Avg_Distance`
- Labels: `0-a (left)`, `1-s (brake)`, `2-d (right)`
- Used **SVM with RBF Kernel** for high-accuracy classification.

---

## 🧰 Technologies Used

| Tool | Purpose |
|------|---------|
| **YOLOv8** | Object Detection |
| **SVM (RBF Kernel)** | Action Classification |
| **OpenCV** | Image Processing |
| **MSS** | Screen Capture |
| **Python** | Programming Language |
| **Keyboard Module** | Simulated Driving Inputs |

---

## 🧪 Testing & Results

### Object Detection
- Over **90% accuracy** in detecting vehicles and pedestrians.

### Action Prediction
- **100% accuracy** on the custom dataset.
- Real-time keypress simulation works with minimal latency.

### Lane Detection
- Reliable in most conditions, some limitations in poor lighting or intersections.

---

## ✅ Conclusion

The ADAS project demonstrates the viability of virtual driver assistance using computer vision and ML in a simulated environment. While it performs well in most scenarios, improvements such as:
- Deep learning-based lane segmentation
- Faster model inference
could enhance its real-world adaptability.

---

💡 Future Work
Implement semantic segmentation for improved lane detection.

Extend to real-world dashcam videos.

Integrate with reinforcement learning for adaptive driving.
