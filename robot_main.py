import cv2
import numpy as np
import time
from ultralytics import YOLO
from cartoon_filter import cartoonify  # Importing your filter

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

print("Starting Autonomous Robot...")
print("Auto-capture in 5 seconds...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create a clean copy for the display (boxes go here)
    display_frame = frame.copy()

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    results = model(frame, conf=0.5, verbose=False)

    best_target = None
    max_score = -1
    scanning_boxes = []

    # 1. LOGIC: Find the best target
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            area = (x2 - x1) * (y2 - y1)
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2
            dist_from_center = np.sqrt((obj_cx - center_x)**2 + (obj_cy - center_y)**2)
            
            score = area / (dist_from_center + 1)

            if score > max_score:
                if best_target:
                    scanning_boxes.append(best_target) 
                max_score = score
                best_target = (x1, y1, x2, y2, label)
            else:
                scanning_boxes.append((x1, y1, x2, y2, label))

    # 2. UI: Draw Scanning Boxes (Blue)
    for (x1, y1, x2, y2, label) in scanning_boxes:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    elapsed = time.time() - start_time
    remaining = max(0, int(5 - elapsed))

    # 3. UI: Draw Locked Box (Red)
    if best_target:
        x1, y1, x2, y2, label = best_target
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(display_frame, f"LOCKED: {label}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    if remaining > 0:
        cv2.putText(display_frame, f"AUTO-CAPTURE: {remaining}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(display_frame, "PROCESSING...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Robot View", display_frame)
    
    # 4. ACTION: Capture Event
    if remaining == 0 and best_target:
        x1, y1, x2, y2, label = best_target
        
        # Grab from original 'frame' (no blue lines), not 'display_frame'
        roi = frame[y1:y2, x1:x2]
        
        cartoon_roi = cartoonify(roi)
        
        side_by_side = np.hstack((roi, cartoon_roi))
        rh, rw, _ = roi.shape
        cv2.putText(side_by_side, "Target", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(side_by_side, "Cartoon", (rw+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Final Robot Output", side_by_side)
        print(f"Captured: {label}")
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()