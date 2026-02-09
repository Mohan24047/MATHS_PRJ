import cv2
import numpy as np

PRESETS = {
    'default': {
        'num_colors': 8,
        'edge_block_size': 11,
        'edge_threshold': 7,
        'smooth_d': 9,
        'smooth_sigma_color': 75,
        'smooth_sigma_space': 75,
        'saturation_boost': 1.3,
        'contrast': 1.2,
        'brightness': 10
    }
}


def cartoonify(img, preset='default', **kwargs):
    """
    Apply cartoon effect to an image.
    
    Args:
        img: Input BGR image (numpy array)
        preset: Style preset ('default', 'anime', 'comic', 'pencil', 'vibrant')
        **kwargs: Override any preset parameter:
            - num_colors (int): Number of colors for quantization (4-16)
            - edge_block_size (int): Block size for edge detection (odd, 7-15)
            - edge_threshold (int): Threshold for edge detection (3-12)
            - smooth_d (int): Bilateral filter diameter (5-15)
            - smooth_sigma_color (int): Bilateral filter sigma color
            - smooth_sigma_space (int): Bilateral filter sigma space
            - saturation_boost (float): Saturation multiplier (1.0-2.0)
            - contrast (float): Contrast multiplier
            - brightness (int): Brightness offset
    
    Returns:
        Cartoonified BGR image
    """
    if img is None or img.size == 0:
        return img
    
    params = PRESETS.get(preset, PRESETS['default']).copy()
    params.update(kwargs)
    
    num_colors = max(2, min(16, params['num_colors']))
    edge_block_size = params['edge_block_size']
    if edge_block_size % 2 == 0:
        edge_block_size += 1
    edge_block_size = max(3, min(21, edge_block_size))
    edge_threshold = max(1, min(20, params['edge_threshold']))
    smooth_d = max(1, min(20, params['smooth_d']))
    smooth_sigma_color = params['smooth_sigma_color']
    smooth_sigma_space = params['smooth_sigma_space']
    saturation_boost = max(0.5, min(2.5, params['saturation_boost']))
    contrast = params['contrast']
    brightness = params['brightness']
    
    enhanced = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    smooth = cv2.bilateralFilter(
        enhanced, 
        d=smooth_d, 
        sigmaColor=smooth_sigma_color, 
        sigmaSpace=smooth_sigma_space
    )
    
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        edge_block_size, 
        edge_threshold
    )
    
    data = np.float32(smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, label, center = cv2.kmeans(
        data, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    center = np.uint8(center)
    quantized = center[label.flatten()]
    quantized = quantized.reshape(img.shape)
    
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    
    return cartoon


def get_available_presets():
    """Return list of available preset names."""
    return list(PRESETS.keys())


def get_preset_params(preset_name):
    """Return parameters for a given preset."""
    return PRESETS.get(preset_name, PRESETS['default']).copy()
