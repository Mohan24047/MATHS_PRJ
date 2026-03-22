import cv2
import numpy as np

PRESETS = {
    'default': {
        'num_colors': 20,           # YOUR TUNED VALUE: Smooths out skin gradients
        'edge_block_size': 9,       # YOUR TUNED VALUE: Catches fine details
        'edge_threshold': 9,        # YOUR TUNED VALUE: Rejects shadow noise
        'smooth_d': 1,              # YOUR TUNED VALUE: Preserves natural texture
        'smooth_sigma_color': 75,   
        'smooth_sigma_space': 75,
        'saturation_boost': 1.2,    # Slight pop to the colors
        'contrast': 1.0,            # Required to prevent HUD crash
        'brightness': 0,            # Required to prevent HUD crash
        'bilateral_iterations': 1   # Reduced since smooth_d is 1
    },
    'bold_comic': {                 
        'num_colors': 12,
        'edge_block_size': 13,
        'edge_threshold': 11,
        'smooth_d': 5,
        'smooth_sigma_color': 100,
        'smooth_sigma_space': 100,
        'saturation_boost': 1.3,
        'contrast': 1.0,            
        'brightness': 0,            
        'bilateral_iterations': 2
    }
}


def cartoonify(img, preset='default', **kwargs):
    if img is None or img.size == 0:
        return img
    
    params = PRESETS.get(preset, PRESETS['default']).copy()
    params.update(kwargs)
    
    num_colors = max(2, min(24, params['num_colors'])) # Increased max to allow your 20
    edge_block_size = params['edge_block_size']
    if edge_block_size % 2 == 0: edge_block_size += 1
    edge_threshold = params['edge_threshold']
    
    # 1. COLOR PREP
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_boost'], 0, 255).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    for _ in range(params['bilateral_iterations']):
        color = cv2.bilateralFilter(
            color, params['smooth_d'], params['smooth_sigma_color'], params['smooth_sigma_space']
        )
    
    # 2. EDGE DETECTION WITH MORPHOLOGY CLEAN-UP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7) 
    
    edges = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        edge_block_size, 
        edge_threshold
    )
    
    # Morphological Closing geometrically erases isolated noise dots
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. K-MEANS QUANTIZATION & MERGE
    data = np.float32(color).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    quantized = np.uint8(center)[label.flatten()].reshape(img.shape)
    
    # Post-Quantization Melt
    quantized = cv2.bilateralFilter(quantized, 7, 50, 50)
    
    return cv2.bitwise_and(quantized, quantized, mask=edges)


def get_available_presets():
    return list(PRESETS.keys())

def get_preset_params(preset_name):
    return PRESETS.get(preset_name, PRESETS['default']).copy()