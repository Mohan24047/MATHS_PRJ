# 🤖 Autonomous Robot — Image Cartoonification

An autonomous vision system that uses **YOLOv8 object detection** and **mathematical image processing** (K-Means clustering, bilateral filtering, adaptive thresholding) to detect real-world objects via webcam and transform them into cartoon-style artwork in real time.

---

## ✨ Features

- **Autonomous Object Detection** — Detects 19 object classes (people, animals, electronics, etc.) using YOLOv8 with a weighted scoring system based on area, centrality, and confidence.
- **Smart Object Prioritization** — A mathematical scoring formula ranks detected objects, with temporal stability tracking across frames to avoid jitter.
- **Cartoonification Pipeline** — Applies a multi-stage image processing pipeline:
  1. **Bilateral Filtering** — Edge-preserving smoothing to flatten textures.
  2. **Adaptive Thresholding** — Generates clean, bold cartoon-style edge maps.
  3. **K-Means Color Quantization** — Reduces the color palette to produce flat, poster-like regions.
  4. **Morphological Cleanup** — Removes isolated noise dots from edge masks.
- **Multiple Style Presets** — Includes `default` and `bold_comic` presets with tunable parameters (color count, edge sensitivity, smoothing intensity, saturation, contrast).
- **Cartoonification Accuracy Metrics** — Computes five quality scores: Texture Removal, Color Simplification, Region Flatness, Edge Enhancement, and Structural Integrity.
- **Live Parameter Tuning** — Interactive trackbar GUI to adjust filter parameters in real time and export optimized preset values.

---

## 📁 Project Structure

```
Maths_Project/
├── robot_main.py        # Main autonomous robot — detection + cartoonification + metrics
├── cartoon_filter.py    # Cartoonification engine (presets, K-Means, bilateral filter)
├── test.py              # Standalone image cartoonifier with Toeplitz smoothing
├── tune_filter.py       # Live trackbar GUI for real-time parameter tuning
├── yolov8n.pt           # Pre-trained YOLOv8 Nano model weights
├── requirements.txt     # Python dependencies
├── runs/                # YOLO detection output directory
└── .venv/               # Python virtual environment
```

---

## 🧮 Mathematical Concepts Used

| Concept | Where Used | Purpose |
|---|---|---|
| **K-Means Clustering** | `cartoon_filter.py`, `test.py` | Color quantization — reduces thousands of colors to K flat regions |
| **Bilateral Filtering** | `cartoon_filter.py`, `test.py` | Edge-preserving smoothing — removes texture while keeping sharp boundaries |
| **Adaptive Thresholding** | `cartoon_filter.py`, `tune_filter.py` | Edge detection — converts grayscale to binary edge masks |
| **Canny Edge Detection** | `test.py`, `robot_main.py` | Gradient-based edge extraction for accuracy measurement |
| **Toeplitz Matrix Convolution** | `test.py` | Separable 1D Gaussian-like filter using `[1,4,6,4,1]` kernel |
| **Morphological Operations** | `cartoon_filter.py` | Closing operation to remove isolated noise pixels from edge masks |
| **Weighted Scoring Function** | `robot_main.py` | Prioritizes detected objects using normalized area, centrality, and confidence |
| **Laplacian Variance** | `robot_main.py` | Measures texture roughness to evaluate smoothing quality |
| **HSV Color Space Manipulation** | `cartoon_filter.py` | Saturation boosting via channel scaling in HSV space |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A webcam (for live detection and tuning)

### Installation

```bash
# Clone or navigate to the project
cd Maths_Project

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Run the Autonomous Robot (Main Demo)
Detects objects via webcam, auto-captures after 5 seconds, and displays all cartoon style presets with accuracy metrics.
```bash
python robot_main.py
```

#### 2. Cartoonify a Static Image
Applies the full pipeline (Toeplitz smoothing → bilateral filter → K-Means → Canny edges) to any image file.
```bash
python test.py
# Enter the path to your image when prompted
```

#### 3. Tune Filter Parameters Live
Opens an interactive window with trackbars to adjust K-Means clusters, edge block size, edge threshold, and blur diameter in real time.
```bash
python tune_filter.py
# Press 'q' to quit and print your tuned values
```

---

## 📊 Accuracy Metrics

The system evaluates cartoonification quality across five dimensions (each weighted equally at 20%):

| Metric | What It Measures |
|---|---|
| **Texture Removal** | How much fine texture/noise was smoothed away (masked Laplacian variance) |
| **Color Simplification** | Reduction in unique color count after K-Means quantization |
| **Region Flatness** | Internal uniformity of color regions (Gaussian blur MSE) |
| **Edge Enhancement** | Whether edge density falls in the ideal cartoon range (2–15%) |
| **Structural Integrity** | Geometric alignment between original contours and cartoon edges (dilated overlap) |

---

## 🛠️ Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing, GUI, video capture |
| `numpy` | Numerical operations and array manipulation |
| `ultralytics` | YOLOv8 object detection model |
| `scikit-learn` | K-Means clustering (used in `test.py`) |

---

## 📝 License

This project was developed as a Mathematics project demonstrating real-world applications of mathematical concepts in computer vision and image processing.
