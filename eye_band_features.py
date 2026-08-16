"""
Eye Band Features - Trich xuat dac diem tu vung mat, tran va chan may
Giup tang do chinh xac khi nhan dien nguoi deo khau trang
"""

import cv2
import numpy as np


def _resolve_frame_and_shape(frame_or_shape):
    if hasattr(frame_or_shape, "shape"):
        frame = frame_or_shape
        frame_shape = frame.shape
    else:
        frame = None
        frame_shape = frame_or_shape
    return frame, frame_shape


def _crop_region(frame, top, left, bottom, right):
    if frame is None:
        return np.array([])
    if bottom <= top or right <= left:
        return np.array([])
    return frame[top:bottom, left:right]


def get_forehead_region(face_box, frame_or_shape, expand_ratio=0.25):
    """
    Trich xuat vung tran (forehead) tu face_box.
    Vung nay nam tren chan may - khong bi che boi khau trang.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame_shape: (height, width, channels) - kich thuoc frame
        expand_ratio: ty le mo rong vung len tren
        
    Returns:
        forehead: anh crop vung tran
        region_box: (top, left, bottom, right) cua vung crop
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    
    face_height = bottom - top
    face_width = right - left
    
    # Vung tran: tu dinh dau xuong
    forehead_top = max(0, top - int(face_height * expand_ratio))
    # Vung duoi: ngay tren chan may
    forehead_bottom = top + int(face_height * 0.18)
    
    # Mo rong sang hai ben
    side_margin = int(face_width * 0.15)
    forehead_left = max(0, left - side_margin)
    forehead_right = min(frame_width, right + side_margin)
    
    # Crop vung tran
    forehead = _crop_region(frame, forehead_top, forehead_left, forehead_bottom, forehead_right)
    
    return forehead, (forehead_top, forehead_left, forehead_bottom, forehead_right)


def get_eyebrow_region(face_box, frame_or_shape):
    """
    Trich xuat vung chan may (eyebrow) tu face_box.
    Chan may co hinh dang va vi tri unique cho moi nguoi.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame_shape: (height, width, channels) - kich thuoc frame
        
    Returns:
        eyebrow: anh crop vung chan may
        region_box: (top, left, bottom, right) cua vung crop
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    
    face_height = bottom - top
    face_width = right - left
    
    # Vung chan may: ngay tren mat
    eyebrow_top = top + int(face_height * 0.15)
    eyebrow_bottom = top + int(face_height * 0.28)
    
    # Mo rong sang hai ben
    side_margin = int(face_width * 0.12)
    eyebrow_left = max(0, left - side_margin)
    eyebrow_right = min(frame_width, right + side_margin)
    
    # Crop vung chan may
    eyebrow = _crop_region(frame, eyebrow_top, eyebrow_left, eyebrow_bottom, eyebrow_right)
    
    return eyebrow, (eyebrow_top, eyebrow_left, eyebrow_bottom, eyebrow_right)


def get_upper_face_region(face_box, frame_or_shape):
    """
    Trich xuat toan bo vung tren cua khuon mat (tran + chan may + mat).
    Day la vung khong bi che boi khau trang.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame_shape: (height, width, channels) - kich thuoc frame
        
    Returns:
        upper_face: anh crop vung tren mat
        region_box: (top, left, bottom, right) cua vung crop
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    
    face_height = bottom - top
    face_width = right - left
    
    # Vung tren: tu dinh dau den ngay duoi mat
    upper_top = max(0, top + int(face_height * 0.04))
    upper_bottom = min(frame_height, top + int(face_height * 0.42))
    
    # Mo rong sang hai ben
    side_margin = int(face_width * 0.12)
    upper_left = max(0, left - side_margin)
    upper_right = min(frame_width, right + side_margin)
    
    # Crop vung tren mat
    upper_face = _crop_region(frame, upper_top, upper_left, upper_bottom, upper_right)
    
    return upper_face, (upper_top, upper_left, upper_bottom, upper_right)


def get_eye_band_region(face_box, frame_or_shape, expand_ratio=0.15):
    """
    Trich xuat vung mat-tran (eye band) tu face_box.
    Vung nay bao gom tran va hai mat - khong bi che boi khau trang.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame_shape: (height, width, channels) - kich thuoc frame
        expand_ratio: ty le mo rong vung len tren
        
    Returns:
        eye_band: anh crop vung mat-tran
        region_box: (top, left, bottom, right) cua vung crop
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    
    # Tinh toan vung mat-tran
    face_height = bottom - top
    face_width = right - left
    
    # Vung tran: tu dinh dau den giua mat
    eye_band_top = max(0, top + int(face_height * 0.08))
    # Vung duoi: ngay duoi mat (tranh phan mieng)
    eye_band_bottom = min(frame_height, top + int(face_height * 0.45))
    
    # Mo rong sang hai ben
    side_margin = int(face_width * 0.1)
    eye_band_left = max(0, left - side_margin)
    eye_band_right = min(frame_width, right + side_margin)
    
    # Crop vung mat-tran
    eye_band = _crop_region(frame, eye_band_top, eye_band_left, eye_band_bottom, eye_band_right)
    
    return eye_band, (eye_band_top, eye_band_left, eye_band_bottom, eye_band_right)


def extract_eye_band_hog(eye_band, cell_size=8, block_size=2, bins=9):
    """
    Trich xuat HOG features tu vung mat-tran.
    
    Args:
        eye_band: anh grayscale vung mat-tran
        cell_size: kich thuoc cell cho HOG
        block_size: kich thuoc block
        bins: so luong bins cho histogram
        
    Returns:
        hog_features: vector dac diem HOG
    """
    if eye_band.size == 0:
        return np.array([])
    
    # Resize ve kich thuoc chuan de dam bao feature vector co dinh
    eye_band = cv2.resize(eye_band, (64, 32))
    
    # Tinh gradient
    gx = cv2.Sobel(eye_band, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(eye_band, cv2.CV_32F, 0, 1, ksize=1)
    
    # Tinh magnitude va goc
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180  # Chuyen ve 0-180
    
    # Tinh HOG thu cong
    h, w = eye_band.shape
    features = []
    
    for y in range(0, h - cell_size + 1, cell_size):
        for x in range(0, w - cell_size + 1, cell_size):
            # Tao histogram cho cell
            hist = np.zeros(bins)
            cell_mag = magnitude[y:y+cell_size, x:x+cell_size]
            cell_ang = angle[y:y+cell_size, x:x+cell_size]
            
            bin_width = 180 / bins
            for i in range(cell_size):
                for j in range(cell_size):
                    bin_idx = int(cell_ang[i, j] / bin_width) % bins
                    hist[bin_idx] += cell_mag[i, j]
            
            # Normalize
            hist = hist / (np.linalg.norm(hist) + 1e-6)
            features.extend(hist)
    
    return np.array(features)


def extract_eye_band_lbp(eye_band, radius=1, points=8):
    """
    Trich xuat LBP (Local Binary Pattern) tu vung mat-tran.
    
    Args:
        eye_band: anh grayscale vung mat-tran
        radius: ban kinh cho LBP
        points: so diem lan can
        
    Returns:
        lbp_hist: histogram LBP
    """
    if eye_band.size == 0:
        return np.array([])
    
    # Resize ve kich thuoc chuan
    eye_band = cv2.resize(eye_band, (64, 32))
    
    h, w = eye_band.shape
    lbp_image = np.zeros((h, w), dtype=np.uint8)
    
    # Tinh LBP thu cong
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = eye_band[y, x]
            code = 0
            
            # 8 diem xung quanh
            neighbors = [
                eye_band[y - radius, x],           # North
                eye_band[y - radius, x + radius], # North-East
                eye_band[y, x + radius],           # East
                eye_band[y + radius, x + radius], # South-East
                eye_band[y + radius, x],           # South
                eye_band[y + radius, x - radius], # South-West
                eye_band[y, x - radius],           # West
                eye_band[y - radius, x - radius], # North-West
            ]
            
            for i, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    code |= (1 << i)
            
            lbp_image[y, x] = code
    
    # Tinh histogram
    hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-6)
    
    return hist


def extract_eye_band_color_hist(eye_band, bins=32):
    """
    Trich xuat histogram mau tu vung mat-tran.
    
    Args:
        eye_band: anh mau BGR vung mat-tran
        bins: so bins cho moi kenh mau
        
    Returns:
        color_hist: histogram mau
    """
    if eye_band.size == 0:
        return np.array([])
    
    # Resize ve kich thuoc chuan
    eye_band = cv2.resize(eye_band, (64, 32))
    
    hist_features = []
    
    # Tinh histogram cho tung kenh BGR
    for i in range(3):
        hist = cv2.calcHist([eye_band], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-6)
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_eye_band_texture(eye_band):
    """
    Trich xuat dac diem texture tu vung mat-tran.
    
    Args:
        eye_band: anh grayscale vung mat-tran
        
    Returns:
        texture_features: vector dac diem texture
    """
    if eye_band.size == 0:
        return np.array([])
    
    # Resize ve kich thuoc chuan
    eye_band = cv2.resize(eye_band, (64, 32))
    
    features = []
    
    # 1. GLCM-like features (tinh toan don gian hoa)
    # Contrast
    gy = cv2.Sobel(eye_band, cv2.CV_64F, 0, 1, ksize=3)
    gx = cv2.Sobel(eye_band, cv2.CV_64F, 1, 0, ksize=3)
    contrast = np.mean(np.abs(gx) + np.abs(gy))
    features.append(contrast)
    
    # Homogeneity
    laplacian = cv2.Laplacian(eye_band, cv2.CV_64F)
    homogeneity = 1.0 / (1.0 + np.var(laplacian))
    features.append(homogeneity)
    
    # Energy
    energy = np.sum(eye_band.astype(np.float32) ** 2) / (eye_band.size * 255**2)
    features.append(energy)
    
    # 2. Edge density
    edges = cv2.Canny(eye_band, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # 3. Statistical features
    features.append(np.mean(eye_band))
    features.append(np.std(eye_band))
    features.append(np.median(eye_band))
    
    # 4. Shape features (ty le khuon mat)
    h, w = eye_band.shape
    features.append(w / h if h > 0 else 0)
    
    return np.array(features)


def extract_all_eye_band_features(face_box, frame, expand_ratio=0.15):
    """
    Trich xuat tat ca dac diem tu vung mat-tran.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        expand_ratio: ty le mo rong vung len tren
        
    Returns:
        combined_features: vector dac diem ket hop
    """
    # Lay vung mat-tran
    eye_band, _ = get_eye_band_region(face_box, frame, expand_ratio)
    
    if eye_band.size == 0:
        return np.array([])
    
    # Chuyen sang grayscale
    gray_eye_band = cv2.cvtColor(eye_band, cv2.COLOR_BGR2GRAY) if len(eye_band.shape) == 3 else eye_band
    
    # Trich xuat cac loai dac diem
    hog_features = extract_eye_band_hog(gray_eye_band)
    lbp_features = extract_eye_band_lbp(gray_eye_band)
    color_features = extract_eye_band_color_hist(eye_band)
    texture_features = extract_eye_band_texture(gray_eye_band)
    
    # Ket hop tat ca dac diem
    combined_features = np.concatenate([
        hog_features,
        lbp_features,
        color_features,
        texture_features
    ])
    
    return combined_features


def get_eye_band_encoding(face_box, frame):
    """
    Lay encoding tu vung mat-tran (wrapper function).
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        
    Returns:
        encoding: vector dac diem
    """
    return extract_all_eye_band_features(face_box, frame)


# ============= NEW: Forehead & Eyebrow Features =============

def extract_region_hog(region, cell_size=8, bins=9):
    """Trich xuat HOG features tu bat ky vung anh nao."""
    if region.size == 0:
        return np.array([])
    
    region = cv2.resize(region, (64, 32))
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    
    h, w = region.shape
    features = []
    
    for y in range(0, h - cell_size + 1, cell_size):
        for x in range(0, w - cell_size + 1, cell_size):
            hist = np.zeros(bins)
            cell_mag = magnitude[y:y+cell_size, x:x+cell_size]
            cell_ang = angle[y:y+cell_size, x:x+cell_size]
            
            bin_width = 180 / bins
            for i in range(cell_size):
                for j in range(cell_size):
                    bin_idx = int(cell_ang[i, j] / bin_width) % bins
                    hist[bin_idx] += cell_mag[i, j]
            
            hist = hist / (np.linalg.norm(hist) + 1e-6)
            features.extend(hist)
    
    return np.array(features)


def extract_region_lbp(region, radius=1):
    """Trich xuat LBP features tu bat ky vung anh nao."""
    if region.size == 0:
        return np.array([])
    
    region = cv2.resize(region, (64, 32))
    h, w = region.shape
    lbp_image = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = region[y, x]
            code = 0
            
            neighbors = [
                region[y - radius, x],
                region[y - radius, x + radius],
                region[y, x + radius],
                region[y + radius, x + radius],
                region[y + radius, x],
                region[y + radius, x - radius],
                region[y, x - radius],
                region[y - radius, x - radius],
            ]
            
            for i, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    code |= (1 << i)
            
            lbp_image[y, x] = code
    
    hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-6)
    
    return hist


def extract_region_color_hist(region, bins=32):
    """Trich xuat histogram mau tu bat ky vung anh nao."""
    if region.size == 0:
        return np.array([])
    
    region = cv2.resize(region, (64, 32))
    
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([region], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-6)
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_region_texture(region):
    """Trich xuat texture features tu bat ky vung anh nao."""
    if region.size == 0:
        return np.array([])
    
    region = cv2.resize(region, (64, 32))
    
    features = []
    
    # Contrast
    gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    contrast = np.mean(np.abs(gx) + np.abs(gy))
    features.append(contrast)
    
    # Homogeneity
    laplacian = cv2.Laplacian(region, cv2.CV_64F)
    homogeneity = 1.0 / (1.0 + np.var(laplacian))
    features.append(homogeneity)
    
    # Energy
    energy = np.sum(region.astype(np.float32) ** 2) / (region.size * 255**2)
    features.append(energy)
    
    # Edge density
    edges = cv2.Canny(region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # Statistical features
    features.append(np.mean(region))
    features.append(np.std(region))
    features.append(np.median(region))
    
    # Shape features
    h, w = region.shape
    features.append(w / h if h > 0 else 0)
    
    return np.array(features)


def extract_forehead_features(face_box, frame):
    """
    Trich xuat tat ca dac diem tu vung tran.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        
    Returns:
        forehead_features: vector dac diem tran
    """
    forehead, _ = get_forehead_region(face_box, frame)
    
    if forehead.size == 0:
        return np.array([])
    
    gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY) if len(forehead.shape) == 3 else forehead
    
    hog = extract_region_hog(gray)
    lbp = extract_region_lbp(gray)
    color = extract_region_color_hist(forehead)
    texture = extract_region_texture(gray)
    
    return np.concatenate([hog, lbp, color, texture])


def extract_eyebrow_features(face_box, frame):
    """
    Trich xuat tat ca dac diem tu vung chan may.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        
    Returns:
        eyebrow_features: vector dac diem chan may
    """
    eyebrow, _ = get_eyebrow_region(face_box, frame)
    
    if eyebrow.size == 0:
        return np.array([])
    
    gray = cv2.cvtColor(eyebrow, cv2.COLOR_BGR2GRAY) if len(eyebrow.shape) == 3 else eyebrow
    
    hog = extract_region_hog(gray)
    lbp = extract_region_lbp(gray)
    color = extract_region_color_hist(eyebrow)
    texture = extract_region_texture(gray)
    
    return np.concatenate([hog, lbp, color, texture])


def extract_upper_face_features(face_box, frame):
    """
    Trich xuat tat ca dac diem tu vung tren mat (tran + chan may + mat).
    Day la vung khong bi che boi khau trang.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        
    Returns:
        combined_features: vector dac diem ket hop tu tran, chan may, mat
    """
    # Trich xuat tung vung
    eye_band_features = extract_all_eye_band_features(face_box, frame)
    forehead_features = extract_forehead_features(face_box, frame)
    eyebrow_features = extract_eyebrow_features(face_box, frame)
    
    # Ket hop tat ca
    all_features = []
    
    if eyebrow_features.size > 0:
        all_features.append(eyebrow_features)
    if eye_band_features.size > 0:
        all_features.append(eye_band_features)
    if forehead_features.size > 0:
        all_features.append(forehead_features)
    
    if not all_features:
        return np.array([])
    
    return np.concatenate(all_features)


def get_upper_face_encoding(face_box, frame):
    """
    Lay encoding tu vung tren mat (tran + chan may + mat).
    Wrapper function cho extract_upper_face_features.
    
    Args:
        face_box: tuple (top, right, bottom, left) - toa do khuon mat
        frame: anh goc BGR
        
    Returns:
        encoding: vector dac diem ket hop
    """
    return extract_upper_face_features(face_box, frame)


def get_left_ear_region(face_box, frame_or_shape, ear_ratio=0.18):
    """
    Trich xuat vung tai trai dua tren face_box.
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    face_height = bottom - top
    face_width = right - left
    
    ear_top = top + int(face_height * 0.18)
    ear_bottom = bottom - int(face_height * 0.18)
    ear_left = max(0, left - int(face_width * ear_ratio))
    ear_right = left + int(face_width * 0.12)
    
    left_ear = _crop_region(frame, ear_top, ear_left, ear_bottom, ear_right)
    return left_ear, (ear_top, ear_left, ear_bottom, ear_right)

def get_right_ear_region(face_box, frame_or_shape, ear_ratio=0.18):
    """
    Trich xuat vung tai phai dua tren face_box.
    """
    frame, frame_shape = _resolve_frame_and_shape(frame_or_shape)
    top, right, bottom, left = face_box
    frame_height, frame_width = frame_shape[:2]
    face_height = bottom - top
    face_width = right - left
    
    ear_top = top + int(face_height * 0.18)
    ear_bottom = bottom - int(face_height * 0.18)
    ear_right = min(frame_width, right + int(face_width * ear_ratio))
    ear_left = right - int(face_width * 0.12)
    
    right_ear = _crop_region(frame, ear_top, ear_left, ear_bottom, ear_right)
    return right_ear, (ear_top, ear_left, ear_bottom, ear_right)

def extract_ear_features(face_box, frame):
    """
    Trich xuat dac trung tu ca hai tai (trai, phai).
    """
    left_ear, _ = get_left_ear_region(face_box, frame)
    right_ear, _ = get_right_ear_region(face_box, frame)
    features = []
    for ear in [left_ear, right_ear]:
        if ear.size == 0:
            continue
        gray = cv2.cvtColor(ear, cv2.COLOR_BGR2GRAY) if len(ear.shape) == 3 else ear
        hog = extract_region_hog(gray)
        lbp = extract_region_lbp(gray)
        color = extract_region_color_hist(ear)
        texture = extract_region_texture(gray)
        features.extend([hog, lbp, color, texture])
    if not features:
        return np.array([])
    return np.concatenate(features)


def extract_full_upper_features(face_box, frame):
    """
    Trich xuat dac trung tong hop: tran, chan may, mat, tai.
    """
    upper_face = extract_upper_face_features(face_box, frame)
    ear_features = extract_ear_features(face_box, frame)
    all_features = []
    if upper_face.size > 0:
        all_features.append(upper_face)
    if ear_features.size > 0:
        all_features.append(ear_features)
    if not all_features:
        return np.array([])
    return np.concatenate(all_features)

def get_full_upper_encoding(face_box, frame):
    """
    Wrapper: lay encoding tong hop vung tren mat va tai.
    """
    return extract_full_upper_features(face_box, frame)
