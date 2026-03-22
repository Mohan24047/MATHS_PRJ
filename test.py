import cv2
import numpy as np
from sklearn.cluster import KMeans

# ===============================
# PARAMETERS
# ===============================
NUM_COLORS = 12
CANNY_LOW = 80
CANNY_HIGH = 160
BILATERAL_D = 12
SIGMA_COLOR = 80
SIGMA_SPACE = 80

# ===============================
# LOAD IMAGE
# ===============================
image_path = input("Enter image path: ")
img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0

# ===============================
# TOEPLITZ FILTER
# ===============================
kernel = np.array([1,4,6,4,1], np.float32)
kernel /= kernel.sum()

def toeplitz_filter(channel):
    temp = cv2.filter2D(channel, -1, kernel.reshape(1,-1))
    temp = cv2.filter2D(temp, -1, kernel.reshape(-1,1))
    return temp

toeplitz_smoothed = np.zeros_like(img_float)
for i in range(3):
    toeplitz_smoothed[:,:,i] = toeplitz_filter(img_float[:,:,i])

# ===============================
# BILATERAL
# ===============================
bilateral = cv2.bilateralFilter(
    (toeplitz_smoothed*255).astype(np.uint8),
    d=BILATERAL_D,
    sigmaColor=SIGMA_COLOR,
    sigmaSpace=SIGMA_SPACE
)
bilateral = bilateral.astype(np.float32) / 255.0

# ===============================
# EDGES
# ===============================
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
edges = cv2.dilate(edges, np.ones((3,3),np.uint8))
edges_inv = 1 - edges/255.0

# ===============================
# KMEANS
# ===============================
pixels = bilateral.reshape((-1,3))
kmeans = KMeans(n_clusters=NUM_COLORS, random_state=42).fit(pixels)
quant = kmeans.cluster_centers_[kmeans.labels_]
quant = quant.reshape(bilateral.shape)

cartoon = quant * edges_inv[:,:,None]
cartoon = np.clip(cartoon,0,1)
cartoon_bgr = cv2.cvtColor((cartoon*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ===============================
# ACCURACY METRICS
# ===============================
orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cart_gray = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2GRAY)

orig_colors = len(np.unique(img.reshape(-1,3), axis=0))
cart_colors = len(np.unique(cartoon_bgr.reshape(-1,3), axis=0))
color_simplification = (1 - cart_colors/orig_colors)*100

orig_edges = cv2.Canny(orig_gray,50,150)
cart_edges = cv2.Canny(cart_gray,50,150)
edge_ratio = (np.sum(cart_edges>0)/np.sum(orig_edges>0))*100

mse = np.mean((orig_gray.astype(float)-cart_gray.astype(float))**2)
structural_change = min(100,(mse/(255**2))*500)

accuracy = (
    color_simplification*0.4 +
    edge_ratio*0.3 +
    structural_change*0.3
)
accuracy = min(100, accuracy)

# ===============================
# CREATE SINGLE DISPLAY PANEL
# ===============================

# Resize to same height
h = min(img.shape[0], cartoon_bgr.shape[0])
img_resized = cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h))
cartoon_resized = cv2.resize(cartoon_bgr, (int(cartoon_bgr.shape[1]*h/cartoon_bgr.shape[0]), h))

combined = np.hstack((img_resized, cartoon_resized))

# Create white space for text
text_panel = np.ones((120, combined.shape[1], 3), dtype=np.uint8) * 255

cv2.putText(text_panel, f"NUM_COLORS: {NUM_COLORS}", (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

cv2.putText(text_panel, f"BILATERAL_D: {BILATERAL_D} | CANNY: {CANNY_LOW}-{CANNY_HIGH}",
            (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

cv2.putText(text_panel, f"Cartoonification Accuracy: {accuracy:.2f}%",
            (20,100), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

final_display = np.vstack((combined, text_panel))

# ===============================
# SHOW SINGLE WINDOW
# ===============================
cv2.imshow("Cartoonification Result", final_display)
cv2.waitKey(0)
cv2.destroyAllWindows()