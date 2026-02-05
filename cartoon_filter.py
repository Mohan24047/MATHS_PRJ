import cv2
import numpy as np

def cartoonify(img):
    # Reduced brightness and contrast boost for a more natural look
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Reduced saturation boost slightly
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3) 
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 11, 7)
    
    data = np.float32(smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8 
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    
    cartoon = cv2.bitwise_and(result, result, mask=edges)
    
    return cartoon