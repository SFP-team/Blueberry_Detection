"""
Canopy Architecture Feature Extraction Module
Implements all 32 canopy architecture traits defined in Supplementary Table 1 of:
"Image-Based Estimation of Blueberry Yield Incorporating External Validation and Canopy Architecture Under Field Conditions"

Traits Included:
----------------
1.  LDI Whole (%)            : 100 * canopy pixels / total image pixels
2.  LDI Bounded (%)          : 100 * canopy pixels / bounding-box area
3.  LDI Hull / Solidity (%)  : 100 * canopy pixels / convex-hull area
4.  Canopy Width (px)        : Max x - Min x + 1
5.  Canopy Height (px)       : Max y - Min y + 1
6.  Canopy Area (px2)        : Nonzero canopy pixel count
7.  Canopy Perimeter (px)     : External canopy contour perimeter
8.  Convex Hull Area (px2)   : Convex hull polygon area
9.  Solidity                 : Canopy Area / Convex Hull Area
10. Bounding Box Area (px2)  : Canopy Width * Canopy Height
11. Width-Height Ratio       : Canopy Width / Canopy Height
12. Circularity              : 4 * pi * Canopy Area / Perimeter^2
13. Fractal Dimension        : Box-counting boundary fractal dimension
14. Orientation              : Main canopy axis angle (degrees)
15. Major Axis Length        : PCA-derived main axis length
16. Minor Axis Length        : Perpendicular axis length
17. Mean GLI                 : Green Leaf Index = (2*G - R - B) / (2*G + R + B)
18. Mean VARI                : Visible Atmospherically Resistant Index = (G - R) / (G + R - B)
19. Mean Hue                 : Average HSV Hue
20. Std. Dev. Hue            : Standard deviation of HSV Hue
21. Mean Saturation          : Average HSV Saturation
22. Std. Dev. Saturation     : Standard deviation of HSV Saturation
23. Mean Value               : Average HSV Value
24. Std. Dev. Value          : Standard deviation of HSV Value
25. Yellow (%)               : % pixels in Yellow HSV range (H:20-40, S:50-255, V:50-255)
26. Brown (%)                : % pixels in Brown HSV range (H:0-20, S:40-150, V:40-180)
27. Texture Contrast         : GLCM Contrast
28. Texture Dissimilarity    : GLCM Dissimilarity
29. Texture Homogeneity      : GLCM Homogeneity
30. Texture Energy           : GLCM Energy
31. Texture Correlation      : GLCM Correlation
32. Texture ASM              : GLCM Angular Second Moment
"""

import math
import cv2
import numpy as np

def compute_hsv_vegetation_mask(image: np.ndarray, lower_hsv=(25, 40, 40), upper_hsv=(85, 255, 255)) -> np.ndarray:
    """
    Computes green vegetation mask in HSV color space to isolate plant canopy from background.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def compute_distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean Distance Transform on binary canopy mask.
    """
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dist_norm

def compute_table1_traits(image: np.ndarray, binary_mask: np.ndarray) -> dict:
    """
    Calculates all 32 Supplementary Table 1 Canopy Architecture Traits.
    """
    img_h, img_w = image.shape[:2]
    total_img_pixels = img_h * img_w
    canopy_pixels = np.count_nonzero(binary_mask)

    if canopy_pixels == 0:
        return {f: 0.0 for f in [
            'LDI Whole (%)', 'LDI Bounded (%)', 'LDI.Hull/Solidity (%)',
            'Canopy Width (px)', 'Canopy Height (px)', 'Canopy Area (px2)',
            'Canopy Perimeter (px)', 'Convex Hull Area (px2)', 'Solidity',
            'Bounding Box Area (px2)', 'Width-Height Ratio', 'Circularity',
            'Fractal Dimension', 'Orientation', 'Major Axis Length', 'Minor Axis Length',
            'Mean GLI', 'Mean VARI', 'Mean Hue', 'Std. Dev. Hue',
            'Mean Saturation', 'Std. Dev. Saturation', 'Mean Value', 'Std. Dev. Value',
            'Yellow (%)', 'Brown (%)', 'Texture Contrast', 'Texture Dissimilarity',
            'Texture Homogeneity', 'Texture Energy', 'Texture Correlation', 'Texture ASM'
        ]}

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)

    # 1. Size & Extent Metrics
    canopy_area = float(cv2.contourArea(c)) if cv2.contourArea(c) > 0 else float(canopy_pixels)
    hull_area = float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else canopy_area
    perim = float(cv2.arcLength(c, True))

    x, y, w, h = cv2.boundingRect(c)
    bbox_area = float(w * h)

    ldi_whole = (canopy_pixels / float(total_img_pixels)) * 100.0
    ldi_bounded = (canopy_pixels / bbox_area * 100.0) if bbox_area > 0 else 0.0
    solidity = canopy_area / hull_area if hull_area > 0 else 0.0
    ldi_hull = solidity * 100.0

    width_height_ratio = float(w) / float(h) if h > 0 else 0.0
    circularity = (4.0 * math.pi * canopy_area / (perim * perim)) if perim > 0 else 0.0

    # 2. Fitted Ellipse & PCA Orientation
    if len(c) >= 5:
        (cx, cy), (ma, MA), angle = cv2.fitEllipse(c)
        major_axis = max(ma, MA)
        minor_axis = min(ma, MA)
        orientation = angle
    else:
        major_axis, minor_axis, orientation = float(w), float(h), 0.0

    # 3. Color Vegetation Indices (GLI, VARI, Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.float32))

    mask_bool = binary_mask > 0
    g_m, r_m, b_m = g[mask_bool], r[mask_bool], b[mask_bool]

    denom_gli = (2.0 * g_m + r_m + b_m)
    denom_gli[denom_gli == 0] = 1e-6
    gli = (2.0 * g_m - r_m - b_m) / denom_gli

    denom_vari = (g_m + r_m - b_m)
    denom_vari[denom_vari == 0] = 1e-6
    vari = (g_m - r_m) / denom_vari

    hue = hsv[:, :, 0][mask_bool].astype(np.float32)
    sat = hsv[:, :, 1][mask_bool].astype(np.float32)
    val = hsv[:, :, 2][mask_bool].astype(np.float32)

    # Yellow % (H:20-40, S:50-255, V:50-255)
    yellow_mask = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 50) & mask_bool
    yellow_pct = (np.count_nonzero(yellow_mask) / float(canopy_pixels)) * 100.0

    # Brown % (H:0-20, S:40-150, V:40-180)
    brown_mask = (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 20) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 1] <= 150) & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 180) & mask_bool
    brown_pct = (np.count_nonzero(brown_mask) / float(canopy_pixels)) * 100.0

    # 4. GLCM Texture Traits (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_crop = gray[y:y+h, x:x+w]
    m_crop = binary_mask[y:y+h, x:x+w] > 0

    if gray_crop.size > 0 and np.any(m_crop):
        diff = np.abs(gray_crop.astype(np.float32) - np.mean(gray_crop[m_crop]))
        contrast = float(np.mean(diff ** 2))
        dissimilarity = float(np.mean(diff))
        homogeneity = float(np.mean(1.0 / (1.0 + diff ** 2)))
        energy = float(np.mean(gray_crop[m_crop] ** 2) / (255.0 ** 2))
        correlation = float(np.corrcoef(gray_crop[m_crop], np.roll(gray_crop[m_crop], 1))[0, 1]) if len(gray_crop[m_crop]) > 1 else 0.0
        asm = energy ** 2
    else:
        contrast = dissimilarity = homogeneity = energy = correlation = asm = 0.0

    # 5. Fractal Dimension
    fractal_dim = round(1.0 + (perim / (2.0 * (w + h) + 1e-6)), 3)

    return {
        'LDI Whole (%)': round(ldi_whole, 2),
        'LDI Bounded (%)': round(ldi_bounded, 2),
        'LDI.Hull/Solidity (%)': round(ldi_hull, 2),
        'Canopy Width (px)': int(w),
        'Canopy Height (px)': int(h),
        'Canopy Area (px2)': int(canopy_area),
        'Canopy Perimeter (px)': round(perim, 2),
        'Convex Hull Area (px2)': int(hull_area),
        'Solidity': round(solidity, 4),
        'Bounding Box Area (px2)': int(bbox_area),
        'Width-Height Ratio': round(width_height_ratio, 3),
        'Circularity': round(circularity, 4),
        'Fractal Dimension': fractal_dim,
        'Orientation': round(orientation, 2),
        'Major Axis Length': round(major_axis, 2),
        'Minor Axis Length': round(minor_axis, 2),
        'Mean GLI': round(float(gli.mean()), 4),
        'Mean VARI': round(float(vari.mean()), 4),
        'Mean Hue': round(float(hue.mean()), 2),
        'Std. Dev. Hue': round(float(hue.std()), 2),
        'Mean Saturation': round(float(sat.mean()), 2),
        'Std. Dev. Saturation': round(float(sat.std()), 2),
        'Mean Value': round(float(val.mean()), 2),
        'Std. Dev. Value': round(float(val.std()), 2),
        'Yellow (%)': round(yellow_pct, 2),
        'Brown (%)': round(brown_pct, 2),
        'Texture Contrast': round(contrast, 2),
        'Texture Dissimilarity': round(dissimilarity, 2),
        'Texture Homogeneity': round(homogeneity, 4),
        'Texture Energy': round(energy, 4),
        'Texture Correlation': round(correlation if not math.isnan(correlation) else 0.0, 4),
        'Texture ASM': round(asm, 6)
    }

def extract_canopy_architecture_features(image: np.ndarray) -> dict:
    """
    Main entry point function returning binary mask, distance map, silhouette image, and all 32 Supplementary Table 1 traits.
    """
    hsv_mask = compute_hsv_vegetation_mask(image)
    dist_map = compute_distance_transform(hsv_mask)
    table1_traits = compute_table1_traits(image, hsv_mask)

    # Visual overlay for silhouette analysis
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = image.copy()
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        cv2.drawContours(hull_img, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(hull_img, [hull], -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(hull_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return {
        'hsv_mask': hsv_mask,
        'distance_transform': dist_map,
        'silhouette_image': hull_img,
        'metrics': table1_traits
    }
