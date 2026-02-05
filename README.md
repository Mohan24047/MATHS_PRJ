# 🤖 Autonomous Cartoonifying Robot

An AI-powered computer vision project that simulates an autonomous robot. It scans for objects using **YOLOv8**, identifies the most "relevant" target based on size and proximity to the center, and "captures" it by applying a custom cartoonization filter.

## 🚀 Features

* **Real-time Object Detection:** Uses the YOLOv8 (Nano) model for high-speed tracking.
* **Target Locking System:** Calculates a "priority score" to lock onto the largest, most centered object in the frame.
* **Cartoonify Pipeline:**
* **Bilateral Filtering:** Smooths textures while keeping edges sharp.
* **K-Means Clustering:** Reduces the color palette for a flat, "painted" look.
* **Adaptive Thresholding:** Generates hand-drawn style outlines.


* **Auto-Capture Timer:** A 5-second countdown to "lock and process" the target.

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/autonomous-cartoon-robot.git
cd autonomous-cartoon-robot

```


2. **Install dependencies:**
```bash
pip install opencv-python numpy ultralytics

```


3. **Download the Model:**
The code will automatically download `yolov8n.pt` on the first run.

---

## 💻 Usage

Run the main script to start the robot's "viewfinder":

```bash
python robot_main.py

```

### How it works:

1. **Scanning:** Blue boxes identify all detected objects.
2. **Locked:** A Red box identifies the "Primary Target."
3. **Capture:** After 5 seconds, the robot crops the target and generates a side-by-side comparison of the raw footage vs. the cartoon output.
4. **Exit:** Press `q` to quit the live feed, or any key to close the final output window.

---

## 🧪 Technical Logic

The robot uses a scoring algorithm to choose its target: Score = Area/(Distance from center + 1)

This ensures that the robot ignores background noise and focuses on the subject directly in front of it.


