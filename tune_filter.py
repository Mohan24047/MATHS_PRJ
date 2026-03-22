import cv2
import numpy as np

def nothing(x):
    pass

# Setup GUI window and sliders
cv2.namedWindow('Live Cartoon Tuning', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Colors (K)', 'Live Cartoon Tuning', 10, 20, nothing)
cv2.createTrackbar('Edge Block', 'Live Cartoon Tuning', 19, 41, nothing)
cv2.createTrackbar('Edge Thresh', 'Live Cartoon Tuning', 11, 30, nothing)
cv2.createTrackbar('Blur (d)', 'Live Cartoon Tuning', 9, 20, nothing)

cap = cv2.VideoCapture(0)

print("Drag the sliders to tune the math in real-time.")
print("Press 'q' to quit and print your final numbers.")

while True:
    ret, img = cap.read()
    if not ret: 
        break

    # Get current slider positions
    k = max(2, cv2.getTrackbarPos('Colors (K)', 'Live Cartoon Tuning'))
    block = max(3, cv2.getTrackbarPos('Edge Block', 'Live Cartoon Tuning'))
    if block % 2 == 0: block += 1
    thresh = cv2.getTrackbarPos('Edge Thresh', 'Live Cartoon Tuning')
    d = max(1, cv2.getTrackbarPos('Blur (d)', 'Live Cartoon Tuning'))

    # 1. Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7) 
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, thresh)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 2. Colors (K-Means)
    color = cv2.bilateralFilter(img, d, 75, 75)
    data = np.float32(color).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    quantized = np.uint8(center)[label.flatten()].reshape(img.shape)

    # 3. Merge
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    cv2.imshow('Live Cartoon Tuning', cartoon)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n--- YOUR PERFECT PRESET NUMBERS ---")
        print(f"'num_colors': {k}")
        print(f"'edge_block_size': {block}")
        print(f"'edge_threshold': {thresh}")
        print(f"'smooth_d': {d}")
        break

cap.release()
cv2.destroyAllWindows()