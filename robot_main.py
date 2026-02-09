import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
from cartoon_filter import cartoonify, get_available_presets, get_preset_params, PRESETS

TARGET_CLASSES = [
    'person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'cell phone', 'bottle', 'cup', 'book', 'laptop', 'keyboard', 'mouse',
    'teddy bear', 'sports ball', 'frisbee'
]

SCORING_WEIGHTS = {
    'area': 0.4,
    'centrality': 0.3,
    'confidence': 0.3
}

STABILITY_FRAMES = 5

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

print("=" * 50)
print("  AUTONOMOUS ROBOT - Image Cartoonification")
print("=" * 50)
print(f"Available presets: {get_available_presets()}")
print(f"Target classes: {len(TARGET_CLASSES)} types")
print("Auto-capture in 5 seconds...")
print("Press 'q' to quit")
print("=" * 50)

start_time = time.time()
object_tracker = defaultdict(int)


def calculate_score(box, frame_shape, center_x, center_y):
    """Calculate weighted score for object prioritization."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])
    
    area = (x2 - x1) * (y2 - y1)
    max_area = frame_shape[0] * frame_shape[1]
    norm_area = area / max_area
    
    obj_cx = (x1 + x2) // 2
    obj_cy = (y1 + y2) // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((obj_cx - center_x)**2 + (obj_cy - center_y)**2)
    norm_centrality = 1 - (dist / max_dist)
    
    score = (
        SCORING_WEIGHTS['area'] * norm_area +
        SCORING_WEIGHTS['centrality'] * norm_centrality +
        SCORING_WEIGHTS['confidence'] * confidence
    )
    
    return score, (x1, y1, x2, y2)


def get_object_key(box):
    """Generate a simple key for tracking object stability."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 20 * 20, (y1 + y2) // 20 * 20
    return f"{int(box.cls[0])}_{cx}_{cy}"


def calculate_quality_metrics(original, cartoon):
    """Calculate cartoonification accuracy metrics comparing original to cartoon."""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cart_gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
    
    orig_laplacian = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    cart_laplacian = cv2.Laplacian(cart_gray, cv2.CV_64F).var()
    texture_removal = max(0, (1 - cart_laplacian / max(orig_laplacian, 1)) * 100)
    
    orig_colors = len(np.unique(original.reshape(-1, 3), axis=0))
    cart_colors = len(np.unique(cartoon.reshape(-1, 3), axis=0))
    color_simplification = max(0, (1 - cart_colors / max(orig_colors, 1)) * 100)
    
    orig_edges = cv2.Canny(orig_gray, 50, 150)
    cart_edges = cv2.Canny(cart_gray, 50, 150)
    orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
    cart_edge_density = np.sum(cart_edges > 0) / cart_edges.size
    edge_enhancement = min(100, (cart_edge_density / max(orig_edge_density, 0.001)) * 50)
    
    h, w = orig_gray.shape
    block_h, block_w = max(1, h//8), max(1, w//8)
    flatness_scores = []
    for i in range(8):
        for j in range(8):
            block = cart_gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                flatness_scores.append(1 - np.std(block) / 128)
    region_flatness = max(0, min(100, np.mean(flatness_scores) * 100))
    
    mse = np.mean((orig_gray.astype(float) - cart_gray.astype(float)) ** 2)
    max_mse = 255 ** 2
    structural_change = min(100, (mse / max_mse) * 500)
    
    cartoonification_accuracy = (
        texture_removal * 0.25 +
        color_simplification * 0.25 +
        region_flatness * 0.25 +
        edge_enhancement * 0.15 +
        structural_change * 0.10
    )
    cartoonification_accuracy = min(100, cartoonification_accuracy)
    
    return {
        'texture_removal': texture_removal,
        'color_simplification': color_simplification,
        'region_flatness': region_flatness,
        'edge_enhancement': edge_enhancement,
        'structural_change': structural_change,
        'cartoonification_accuracy': cartoonification_accuracy
    }



def create_all_styles_grid(roi):
    """Apply all presets and create a grid display with parameters shown on image."""
    presets = get_available_presets()
    results = []
    metrics = {}
    
    img_h, img_w = 300, 400
    info_h = 280
    total_h = img_h + info_h
    
    roi_resized = cv2.resize(roi, (img_w, img_h))
    
    original_panel = np.zeros((total_h, img_w, 3), dtype=np.uint8)
    original_panel[:img_h] = roi_resized
    original_panel[img_h:] = (40, 40, 40)
    cv2.putText(original_panel, "ORIGINAL", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(original_panel, "No processing applied", (15, img_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    results.append(original_panel)
    
    for preset in presets:
        params = get_preset_params(preset)
        cartoon = cartoonify(roi, preset=preset)
        cartoon_resized = cv2.resize(cartoon, (img_w, img_h))
        
        quality = calculate_quality_metrics(roi, cartoon)
        metrics[preset] = quality
        
        panel = np.zeros((total_h, img_w, 3), dtype=np.uint8)
        panel[:img_h] = cartoon_resized
        panel[img_h:] = (40, 40, 40)
        
        cv2.putText(panel, preset.upper(), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        y = img_h + 35
        cv2.putText(panel, "PARAMETERS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y += 28
        cv2.putText(panel, f"Colors (K): {params['num_colors']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Edge Block: {params['edge_block_size']}", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
        cv2.putText(panel, f"Edge Threshold: {params['edge_threshold']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Smooth: {params['smooth_d']}", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
        cv2.putText(panel, f"Sigma Color: {params['smooth_sigma_color']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Sigma Space: {params['smooth_sigma_space']}", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
        cv2.putText(panel, f"Saturation: {params['saturation_boost']}x", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Contrast: {params['contrast']}x", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Brightness: {params['brightness']}", (290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 35
        cv2.putText(panel, "ACCURACY:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        y += 28
        cv2.putText(panel, f"Texture: {quality['texture_removal']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Color: {quality['color_simplification']:.1f}%", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
        cv2.putText(panel, f"Flatness: {quality['region_flatness']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Edge: {quality['edge_enhancement']:.1f}%", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
        cv2.putText(panel, f"Structure: {quality['structural_change']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 30
        cv2.putText(panel, f"TOTAL ACCURACY: {quality['cartoonification_accuracy']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        results.append(panel)
    
    grid = np.hstack(results)
    
    return grid, metrics



def print_style_report(metrics):
    """Print detailed report for each style."""
    print("\n" + "=" * 80)
    print("  CARTOONIFICATION ACCURACY REPORT")
    print("=" * 80)
    
    print("\n📋 PARAMETER DEFINITIONS:")
    print("─" * 80)
    print("  • num_colors (K)      : Number of colors in K-means quantization (fewer = flatter)")
    print("  • edge_block_size     : Block size for adaptive threshold (larger = thicker edges)")
    print("  • edge_threshold (C)  : Threshold constant for edge detection (lower = more edges)")
    print("  • smooth_d            : Bilateral filter diameter (higher = more smoothing)")
    print("  • smooth_sigma_color  : Color similarity range (higher = blend similar colors)")
    print("  • smooth_sigma_space  : Spatial range (higher = consider distant pixels)")
    print("  • saturation_boost    : Color vibrancy multiplier (1.0 = original, 2.0 = double)")
    print("  • contrast            : Contrast multiplier (1.0 = original)")
    print("  • brightness          : Brightness offset added to pixels")
    print("─" * 80)
    
    print("\n📊 CARTOONIFICATION METRICS EXPLAINED:")
    print("─" * 80)
    print("  • Texture Removal      : How much fine texture/noise was removed (higher = smoother)")
    print("  • Color Simplification : Reduction in unique colors (higher = flatter cartoon look)")
    print("  • Region Flatness      : Uniformity of color within regions (higher = more cartoon-like)")
    print("  • Edge Enhancement     : Strength of cartoon edges (higher = bolder outlines)")
    print("  • Structural Change    : How different from original photo (higher = more transformed)")
    print("─" * 80)
    
    for preset in get_available_presets():
        params = get_preset_params(preset)
        quality = metrics[preset]
        
        print(f"\n┌{'─' * 78}┐")
        print(f"│  🎨 {preset.upper():^70}  │")
        print(f"├{'─' * 78}┤")
        print(f"│  PARAMETERS:                                                                 │")
        print(f"│    • num_colors (K-means clusters)     = {params['num_colors']:>3}   (range: 4-16)              │")
        print(f"│    • edge_block_size (threshold block) = {params['edge_block_size']:>3}   (range: 7-15, odd)         │")
        print(f"│    • edge_threshold (sensitivity)      = {params['edge_threshold']:>3}   (range: 3-12)              │")
        print(f"│    • smooth_d (filter diameter)        = {params['smooth_d']:>3}   (range: 5-15)              │")
        print(f"│    • smooth_sigma_color                = {params['smooth_sigma_color']:>3}   (higher = blend colors)   │")
        print(f"│    • smooth_sigma_space                = {params['smooth_sigma_space']:>3}   (higher = blend areas)    │")
        print(f"│    • saturation_boost                  = {params['saturation_boost']:>3.1f}x  (1.0 = no change)         │")
        print(f"│    • contrast                          = {params['contrast']:>3.1f}x  (1.0 = no change)         │")
        print(f"│    • brightness                        = {params['brightness']:>3}   (offset added)            │")
        print(f"├{'─' * 78}┤")
        print(f"│  CARTOONIFICATION ACCURACY:                                                  │")
        print(f"│    • Texture Removal:      {quality['texture_removal']:>5.1f}%  (fine detail removed)              │")
        print(f"│    • Color Simplification: {quality['color_simplification']:>5.1f}%  (palette reduced)                │")
        print(f"│    • Region Flatness:      {quality['region_flatness']:>5.1f}%  (uniform color areas)              │")
        print(f"│    • Edge Enhancement:     {quality['edge_enhancement']:>5.1f}%  (bold outlines)                   │")
        print(f"│    • Structural Change:    {quality['structural_change']:>5.1f}%  (transformation from photo)       │")
        print(f"│                                                                              │")
        print(f"│    ★ CARTOONIFICATION ACCURACY: {quality['cartoonification_accuracy']:>5.1f}%                                  │")
        print(f"└{'─' * 78}┘")
    
    best = max(metrics.keys(), key=lambda k: metrics[k]['cartoonification_accuracy'])
    print(f"\n🏆 BEST CARTOONIFICATION: {best.upper()} ({metrics[best]['cartoonification_accuracy']:.1f}%)")
    print("=" * 80)


def run_robot():
    cap = cv2.VideoCapture(0)
    print("=" * 50)
    print("  AUTONOMOUS ROBOT - Image Cartoonification")
    print("=" * 50)
    print(f"Available presets: {get_available_presets()}")
    print(f"Target classes: {len(TARGET_CLASSES)} types")
    print("Auto-capture in 5 seconds...")
    print("Press 'q' to quit")
    print("=" * 50)

    start_time = time.time()
    object_tracker = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        results = model(frame, conf=0.5, verbose=False)

        best_target = None
        max_score = -1
        scanning_boxes = []
        current_objects = set()

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                
                if label not in TARGET_CLASSES:
                    continue
                
                obj_key = get_object_key(box)
                current_objects.add(obj_key)
                object_tracker[obj_key] += 1
                
                score, coords = calculate_score(box, frame.shape, center_x, center_y)
                
                stability_bonus = min(object_tracker[obj_key] / STABILITY_FRAMES, 1.0) * 0.1
                score += stability_bonus
                
                x1, y1, x2, y2 = coords

                if score > max_score:
                    if best_target:
                        scanning_boxes.append(best_target)
                    max_score = score
                    best_target = (x1, y1, x2, y2, label, score)
                else:
                    scanning_boxes.append((x1, y1, x2, y2, label, score))

        keys_to_remove = [k for k in object_tracker if k not in current_objects]
        for k in keys_to_remove:
            del object_tracker[k]

        for (x1, y1, x2, y2, label, score) in scanning_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{label} ({score:.2f})", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        elapsed = time.time() - start_time
        remaining = max(0, int(5 - elapsed))

        if best_target:
            x1, y1, x2, y2, label, score = best_target
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(display_frame, f"LOCKED: {label} ({score:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if remaining > 0:
            cv2.putText(display_frame, f"AUTO-CAPTURE: {remaining}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(display_frame, "PROCESSING...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Robot View", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if remaining == 0 and best_target:
            x1, y1, x2, y2, label, score = best_target
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                grid, metrics = create_all_styles_grid(roi)
                
                cv2.imshow("All Cartoon Styles", grid)
                print(f"\nCaptured: {label} | Detection Score: {score:.3f}")
                
                print_style_report(metrics)
                
                cv2.waitKey(0)
                break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot()
