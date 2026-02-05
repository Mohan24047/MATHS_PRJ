# 🤖 Autonomous Target-Locking Cartoon Robot

An intelligent computer vision system that simulates an autonomous robot. Using **YOLOv8** and **OpenCV**, the robot scans the environment, identifies a priority target based on a center-weighted scoring algorithm, and applies a multi-stage **K-Means Clustering** cartoon filter to the captured subject.

## 🌟 Features

* **Dynamic Target Selection:** Uses a mathematical scoring system to "lock" onto the most relevant object (prioritizing size and proximity to center).
* **Neural Network Integration:** Powered by **YOLOv8** for real-time, multi-class object detection.
* **Artistic Image Pipeline:** * **Bilateral Filtering:** For edge-preserving noise reduction.
* **K-Means Quantization:** Reduces image color depth for a "flat" painted aesthetic.
* **Adaptive Thresholding:** Generates high-contrast ink outlines.


* **Interactive HUD:** Real-time visual feedback with scanning boxes, locking indicators, and an auto-capture countdown.

---

## 🛠️ Technical Deep Dive

### 1. The "Lock-On" Logic

The robot chooses its primary target using a specific score calculation to ensure it doesn't get distracted by small background objects:  **Score = Area/(Distance_from_center + 1)**

This formula ensures that an object twice as large as another is prioritized, but only if it is also relatively centered in the robot's field of view.

### 2. The Cartoonify Pipeline

The `cartoonify(img)` function follows a sophisticated computer vision workflow:

1. **Saturation Boost:** Converts to HSV space to increase color vibrancy.
2. **Smoothing:** Uses a Bilateral Filter to flatten textures while keeping boundaries sharp.
3. **Quantization:** Uses **K-Means Clustering** () to group pixels into a limited palette.
4. **Edge Masking:** Applies an adaptive mean threshold to create the "comic book" outlines.
5. **Composition:** Merges the quantized colors with the edge mask.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Webcam

### Installation

1. **Clone the repo**
```bash
git clone https://github.com/Mohan24047/MATHS_PRJ.git
cd MATHS_PRJ

```


2. **Install Dependencies**
```bash
pip install opencv-python numpy ultralytics

```



### Running the Robot

```bash
python robot_main.py

```

* **Scanning Phase:** The robot identifies all objects (Blue Boxes).
* **Locked Phase:** The highest-scoring object is highlighted (Red Box).
* **Capture:** After 5 seconds, the robot captures the ROI (Region of Interest) and displays the side-by-side cartoon result.
* **Exit:** Press `q` to quit or any key to close the final output.

---

