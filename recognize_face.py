import math
import os
import pickle
import time
from collections import Counter, deque

import cv2
import face_recognition
import numpy as np

from camera_utils import create_camera
from eye_band_features import get_full_upper_encoding
from svm_model import load_state_svm, predict_face_state_svm_details
from train_model import save_model

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
BADGE_BG_COLOR = (26, 92, 74)
BADGE_TEXT_COLOR = (255, 255, 255)
MULTI_FACE_BLOCK_SECONDS = 1.0
SINGLE_FACE_STABLE_SECONDS = 0.35
MIN_FACE_BRIGHTNESS = 50.0
MIN_FACE_SHARPNESS = 20.0
REQUIRED_CONSISTENT_FRAMES = 4
MASK_REQUIRED_CONSISTENT_FRAMES = 4
OCCLUDED_COMBINED_DISTANCE_THRESHOLD = 160.0
MASK_GLASSES_COMBINED_DISTANCE_THRESHOLD = 210.0
OCCLUDED_CONFUSION_MARGIN = 15.0
MIN_FACE_WIDTH_RATIO = 0.21
MAX_FACE_WIDTH_RATIO = 0.33
FACE_WIDTH_MIN_CLEAR_RATIO = 0.23
FACE_WIDTH_MAX_CLEAR_RATIO = 0.31
LOWER_FACE_OCCLUSION_RATIO = 0.42
EYE_GLARE_BRIGHT_RATIO = 0.10
MASK_DISTANCE_THRESHOLD_BONUS = 0.04
MASK_SVM_PROBABILITY_BONUS = 0.06
MASK_EDGE_RATIO = 1.1
MASK_TEXTURE_RATIO = 1.1
MASK_CENTER_BRIGHT_RATIO = 0.16
MASK_CENTER_STD_MAX = 42.0
MASK_CENTER_EDGE_MAX = 0.13
MASK_CENTER_LIGHT_RATIO = 0.18
MASK_CENTER_LOW_SAT_RATIO = 0.20
STATE_NORMAL_OVERRIDE_PROBABILITY = 0.40
STATE_NORMAL_CLEAR_PROBABILITY = 0.75
STATE_ACCESSORY_SUSPECT_SCORE = 0.25
STATE_STRONG_ACCESSORY_SCORE = 0.65
STATE_MEDIUM_MASK_SCORE = 0.52
STATE_MEDIUM_GLASSES_SCORE = 0.46
STATE_EYE_EDGE_RATIO_THRESHOLD = 0.095
STATE_EYE_GLARE_RATIO_THRESHOLD = 0.04
STATE_EYE_SIDE_EDGE_MIN_RATIO = 0.040
STATE_EYE_EDGE_BALANCE_MAX_RATIO = 3.0
STATE_EYE_BRIDGE_EDGE_RATIO_THRESHOLD = 0.060
STATE_EYE_STRONG_EDGE_RATIO_THRESHOLD = 0.105
STATE_EYE_STRONG_BRIDGE_EDGE_RATIO_THRESHOLD = 0.100
STATE_EYE_STRONG_SIDE_EDGE_MIN_RATIO = 0.075
STATE_WINDOW_SIZE = 7
STATE_MIN_VOTES = 5
FACE_POSE_MAX_ROLL_DEGREES = 12.0
FACE_POSE_MAX_YAW_RATIO = 0.20
FACE_POSE_MAX_NOSE_OFFSET_RATIO = 0.14
FRAME_PROCESS_SCALE = 0.5
YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.60
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
DETECTION_BACKENDS = (
    {"model": "hog", "upsample": 0},
    {"model": "hog", "upsample": 1},
)
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_FACE_SIZE = (40, 40)
TRACK_REFRESH_SECONDS = 1.0
TRACK_TEMPLATE_MATCH_THRESHOLD = 0.45
TRACK_SEARCH_MARGIN_RATIO = 0.65
MAX_FACE_CENTER_SHIFT_RATIO = 0.10
TEMPORAL_WINDOW_SIZE = 7
TEMPORAL_MIN_VOTES = 4
ENABLE_ONLINE_LEARNING = os.getenv("LOCKER_ENABLE_ONLINE_LEARNING", "").strip().lower() in {"1", "true", "yes", "on"}

_HAAR_CASCADE = None
_YUNET_DETECTOR = None

def enhance_lighting(rgb_image):
    """Can bang sang thich ung (CLAHE) giup on dinh nhan dien duoi moi dieu kien anh sang."""
    try:
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception:
        return rgb_image


def calculate_normalized_confidence(distance, threshold):
    """
    Dong bo hoa % Confidence.
    Khoang cach == 0 -> 100%
    Khoang cach == threshold -> 58% (Review zone)
    Khoang cach == threshold * 0.76 -> ~68% (Pass zone)
    """
    conf = 100.0 - (distance / max(threshold, 1e-6)) * 42.0
    return max(0.0, min(100.0, conf))


def load_encodings(data_file="encodings.pickle"):
    if not os.path.exists(data_file):
        return {"encodings": [], "names": []}

    with open(data_file, "rb") as file_obj:
        return pickle.load(file_obj)


def get_guide_rect(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    guide_width = int(frame_width * 0.40)
    guide_height = int(frame_height * 0.55)
    left = (frame_width - guide_width) // 2
    top = (frame_height - guide_height) // 2
    return left, top, left + guide_width, top + guide_height


def is_face_inside_guide(face_box, guide_rect):
    top, right, bottom, left = face_box
    guide_left, guide_top, guide_right, guide_bottom = guide_rect
    face_center_x = (left + right) // 2
    face_center_y = (top + bottom) // 2
    return (
        guide_left <= face_center_x <= guide_right
        and guide_top <= face_center_y <= guide_bottom
    )


def apply_focus_overlay(frame, guide_rect, is_aligned=False):
    # The kiosk UI adds the modern scan corners; keep the raw camera feed clean here.
    return frame


def measure_face_quality(frame, face_box):
    top, right, bottom, left = face_box
    face_crop = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if face_crop.size == 0:
        return 0.0, 0.0

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray_face.mean())
    sharpness = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    return brightness, sharpness


def get_face_width_ratio(frame_shape, face_box):
    _top, right, _bottom, left = face_box
    frame_width = frame_shape[1]
    if frame_width <= 0:
        return 0.0
    return max(0.0, (right - left) / frame_width)


def is_face_frontal_enough(frame, face_box):
    top, right, bottom, left = face_box
    face_width = max(1, right - left)

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_box])
    except Exception:
        return True, {}

    if not landmarks_list:
        return True, {}

    landmarks = landmarks_list[0]
    left_eye = landmarks.get("left_eye") or []
    right_eye = landmarks.get("right_eye") or []
    nose_tip = landmarks.get("nose_tip") or []
    if len(left_eye) < 2 or len(right_eye) < 2 or not nose_tip:
        return True, {}

    left_eye_center = (
        sum(point[0] for point in left_eye) / len(left_eye),
        sum(point[1] for point in left_eye) / len(left_eye),
    )
    right_eye_center = (
        sum(point[0] for point in right_eye) / len(right_eye),
        sum(point[1] for point in right_eye) / len(right_eye),
    )
    nose_center_x = sum(point[0] for point in nose_tip) / len(nose_tip)
    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    inter_eye_distance = max(1.0, abs(right_eye_center[0] - left_eye_center[0]))
    roll_degrees = math.degrees(
        math.atan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0],
        )
    )
    yaw_ratio = abs(nose_center_x - eye_center_x) / inter_eye_distance
    nose_offset_ratio = abs(nose_center_x - ((left + right) / 2.0)) / face_width
    metrics = {
        "roll_degrees": roll_degrees,
        "yaw_ratio": yaw_ratio,
        "nose_offset_ratio": nose_offset_ratio,
    }
    is_frontal = (
        abs(roll_degrees) <= FACE_POSE_MAX_ROLL_DEGREES
        and yaw_ratio <= FACE_POSE_MAX_YAW_RATIO
        and nose_offset_ratio <= FACE_POSE_MAX_NOSE_OFFSET_RATIO
    )
    return is_frontal, metrics


def has_lower_face_mask_signal(face_crop):
    height, width = face_crop.shape[:2]
    if height < 30 or width < 30:
        return False

    lower_region = face_crop[
        int(height * 0.48) : int(height * 0.86),
        int(width * 0.20) : int(width * 0.80),
    ]
    upper_region = face_crop[
        int(height * 0.16) : int(height * 0.40),
        int(width * 0.25) : int(width * 0.75),
    ]
    if lower_region.size == 0 or upper_region.size == 0:
        return False

    lower_hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
    upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
    lower_saturation = lower_hsv[:, :, 1]
    lower_value = lower_hsv[:, :, 2]
    upper_value_mean = float(upper_hsv[:, :, 2].mean())
    lower_value_mean = float(lower_value.mean())
    lower_gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
    lower_mean = lower_region.reshape(-1, 3).mean(axis=0)
    upper_mean = upper_region.reshape(-1, 3).mean(axis=0)

    light_low_saturation_ratio = float(
        ((lower_saturation < 65) & (lower_value > 120)).mean()
    )
    dark_ratio = float((lower_value < 65).mean())
    color_delta = float(np.abs(lower_mean - upper_mean).mean())
    lower_std = float(lower_gray.std())

    white_mask = (
        light_low_saturation_ratio >= 0.30
        and color_delta >= 35.0
        and lower_value_mean >= upper_value_mean + 25.0
    )
    dark_mask = (
        dark_ratio >= 0.45
        and color_delta >= 35.0
        and lower_value_mean + 25.0 <= upper_value_mean
    )
    color_mask = (
        color_delta >= 55.0
        and lower_std <= 65.0
        and abs(lower_value_mean - upper_value_mean) >= 25.0
    )
    return white_mask or dark_mask or color_mask


def classify_face_state(frame, face_box, state_svm_model):
    """
    Classify face into 'normal', 'mask', 'glasses', 'mask_glasses' using SVM.
    """
    top, right, bottom, left = face_box
    face_crop = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if face_crop.size == 0:
        return "normal"

    predicted_state, probability_map = predict_face_state_svm_details(face_crop, state_svm_model)
    
    mask_prob = float(probability_map.get("mask", 0.0))
    glasses_prob = float(probability_map.get("glasses", 0.0))
    mg_prob = float(probability_map.get("mask_glasses", 0.0))
    normal_prob = float(probability_map.get("normal", 0.0))
    
    mask_score = mask_prob + mg_prob
    glasses_score = glasses_prob + mg_prob
    has_mask = has_lower_face_mask_signal(face_crop)
    
    # 1. Phat hien KINH bang Heuristic vung mat
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    height, width = gray_face.shape[:2]
    has_glasses = False
    has_strong_glasses_signal = False
    
    if height >= 20 and width >= 20:
        eye_region = gray_face[int(height * 0.20) : int(height * 0.58), int(width * 0.08) : int(width * 0.92)]
        if eye_region.size > 0:
            edges = cv2.Canny(eye_region, 60, 140)
            edge_ratio = float((edges > 0).mean())
            glare_ratio = float((eye_region >= 230).mean())
            half_width = max(1, eye_region.shape[1] // 2)
            left_edge_ratio = float((edges[:, :half_width] > 0).mean())
            right_edge_ratio = float((edges[:, half_width:] > 0).mean())
            weaker_edge_ratio = min(left_edge_ratio, right_edge_ratio)
            stronger_edge_ratio = max(left_edge_ratio, right_edge_ratio)
            edge_balance = stronger_edge_ratio / max(weaker_edge_ratio, 1e-6)
            bridge_half_width = max(1, int(eye_region.shape[1] * 0.12))
            bridge_left = max(0, half_width - bridge_half_width)
            bridge_right = min(eye_region.shape[1], half_width + bridge_half_width)
            bridge_region = edges[:, bridge_left:bridge_right]
            bridge_edge_ratio = (
                float((bridge_region > 0).mean()) if bridge_region.size > 0 else 0.0
            )
            has_balanced_edges = (
                edge_ratio >= STATE_EYE_EDGE_RATIO_THRESHOLD
                and weaker_edge_ratio >= STATE_EYE_SIDE_EDGE_MIN_RATIO
                and edge_balance <= STATE_EYE_EDGE_BALANCE_MAX_RATIO
            )
            has_bridge_edges = bridge_edge_ratio >= STATE_EYE_BRIDGE_EDGE_RATIO_THRESHOLD
            if has_balanced_edges or has_bridge_edges or glare_ratio >= STATE_EYE_GLARE_RATIO_THRESHOLD:
                has_glasses = True
            has_strong_glasses_signal = (
                edge_ratio >= STATE_EYE_STRONG_EDGE_RATIO_THRESHOLD
                and bridge_edge_ratio >= STATE_EYE_STRONG_BRIDGE_EDGE_RATIO_THRESHOLD
                and weaker_edge_ratio >= STATE_EYE_STRONG_SIDE_EDGE_MIN_RATIO
            )
                
    # Mat tran that thuong van co mask/glasses score nho do anh sang va toc; chi chan khi tin hieu phu kien du manh.
    if (
        normal_prob >= STATE_NORMAL_CLEAR_PROBABILITY
        and max(mask_score, glasses_score) < STATE_ACCESSORY_SUSPECT_SCORE
        and not has_mask
        and not has_strong_glasses_signal
    ):
        return "normal"

    is_mask = (
        has_mask
        or mask_score >= STATE_STRONG_ACCESSORY_SCORE
        or (
            predicted_state in {"mask", "mask_glasses"}
            and normal_prob < STATE_NORMAL_OVERRIDE_PROBABILITY
            and mask_score >= STATE_MEDIUM_MASK_SCORE
        )
    )
    is_glasses = (
        has_strong_glasses_signal
        or
        glasses_score >= STATE_STRONG_ACCESSORY_SCORE
        or (has_glasses and glasses_score >= STATE_ACCESSORY_SUSPECT_SCORE)
        or (
            predicted_state in {"glasses", "mask_glasses"}
            and glasses_score >= STATE_ACCESSORY_SUSPECT_SCORE
            and normal_prob < STATE_NORMAL_CLEAR_PROBABILITY
        )
        or (
            predicted_state in {"glasses", "mask_glasses"}
            and has_glasses
            and normal_prob < STATE_NORMAL_OVERRIDE_PROBABILITY
            and glasses_score >= STATE_MEDIUM_GLASSES_SCORE
        )
    )

    if is_mask and is_glasses:
        if has_glasses and mask_score < STATE_MEDIUM_MASK_SCORE:
            return "glasses"
        if not has_glasses and glasses_score < STATE_MEDIUM_GLASSES_SCORE:
            return "mask"
        return "mask_glasses"
        
    if is_mask:
        return "mask"
    if is_glasses:
        return "glasses"
        
    return "normal"


def _get_yunet_detector():
    global _YUNET_DETECTOR
    if _YUNET_DETECTOR is not None:
        return _YUNET_DETECTOR

    if not os.path.exists(YUNET_MODEL_FILE):
        _YUNET_DETECTOR = False
        return _YUNET_DETECTOR

    try:
        _YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            YUNET_MODEL_FILE,
            "",
            YUNET_INPUT_SIZE,
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
    except Exception:
        _YUNET_DETECTOR = False
    return _YUNET_DETECTOR


def _detect_face_boxes_yunet(rgb_frame):
    detector = _get_yunet_detector()
    if not detector:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    detector.setInputSize((frame_width, frame_height))
    _retval, detections = detector.detect(rgb_frame)
    if detections is None or len(detections) == 0:
        return []

    boxes = []
    for detection in detections:
        x, y, w, h = detection[:4]
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(frame_width, int(round(x + w)))
        bottom = min(frame_height, int(round(y + h)))
        if right > left and bottom > top:
            boxes.append((top, right, bottom, left))

    boxes.sort(key=lambda box: (box[2] - box[0]) * (box[1] - box[3]), reverse=True)
    return boxes


def detect_face_boxes(rgb_frame, prefer_yunet=False):
    last_error = None
    yunet_available = _get_yunet_detector()

    if yunet_available:
        yunet_boxes = _detect_face_boxes_yunet(rgb_frame)
        if yunet_boxes:
            return yunet_boxes, "YUNET"
        last_error = None

    if not prefer_yunet:
        for backend in DETECTION_BACKENDS:
            try:
                boxes = face_recognition.face_locations(
                    rgb_frame,
                    model=backend["model"],
                    number_of_times_to_upsample=backend["upsample"],
                )
            except Exception as exc:
                last_error = exc
                continue
            if boxes:
                return boxes, "HOG"

    if last_error is not None:
        return [], "Detect unavailable"
    return [], "No face"


class FaceRecognitionSession:
    def __init__(
        self,
        data_file="encodings.pickle",
        threshold=0.42,
        timeout_seconds=10,
        min_display_seconds=0,
        mode='take',
    ):
        self.data_file = data_file
        data = load_encodings(data_file)
        self.known_encodings = data.get("encodings", [])
        self.known_names = data.get("names", [])
        self.base_encodings = []
        self.base_names = []
        self.combined_encodings = []
        self.combined_names = []
        for name, encoding in zip(self.known_names, self.known_encodings):
            vector = np.asarray(encoding, dtype=np.float64)
            if len(vector) > 128:
                self.combined_encodings.append(vector)
                self.combined_names.append(name)
            elif len(vector) >= 128:
                self.base_encodings.append(vector[:128])
                self.base_names.append(name)
        self.state_svm_model = load_state_svm()
        self.threshold = threshold
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.min_display_seconds = min_display_seconds

        self.prev_time = time.time()
        self.process_this_frame = True
        self.boxes = []
        self.names = []
        self.confidences = []
        self.detected_frame_count = 0
        self.recognition_attempt_count = 0
        self.recognized_name = None
        self.recognized_confidence = 0.0
        self.face_detected_at = None
        self.single_face_started_at = None
        self.multi_face_blocked_until = 0.0
        self.status_text = "Dua 1 khuon mat vao giua khung"
        self.status_color = (0, 255, 255)
        self.backend_text = "Khoi tao"
        self.show_identity_label = False
        self.confirmed_match_frames = 0
        self.last_predicted_name = None
        self.mask_detected_frames = 0
        self.summary_text = ""
        self._last_console_summary = None
        self.done = False
        self.result = None
        self.added_online = 0
        self.tracked_box = None
        self.tracker_template = None
        self.tracker_lost_frames = 0
        self.last_detection_time = 0.0
        self.prediction_window = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        self.encoding_window = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        self.locked_face_class = None
        self.face_state_window = deque(maxlen=STATE_WINDOW_SIZE)
        self.stable_face_class = "normal"
        self.previous_face_box = None
        self.distance_guidance = None
        self.last_quality_metrics = {}
        self.last_prediction_metrics = {}

    @staticmethod
    def _clip_box(box, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        top, right, bottom, left = box
        top = max(0, min(int(top), frame_height - 1))
        bottom = max(top + 1, min(int(bottom), frame_height))
        left = max(0, min(int(left), frame_width - 1))
        right = max(left + 1, min(int(right), frame_width))
        return top, right, bottom, left

    @staticmethod
    def _scale_box_to_small(box):
        top, right, bottom, left = box
        return (
            int(round(top * FRAME_PROCESS_SCALE)),
            int(round(right * FRAME_PROCESS_SCALE)),
            int(round(bottom * FRAME_PROCESS_SCALE)),
            int(round(left * FRAME_PROCESS_SCALE)),
        )

    def _reset_tracker(self):
        self.tracked_box = None
        self.tracker_template = None
        self.tracker_lost_frames = 0

    def _reset_temporal_smoothing(self):
        self.prediction_window.clear()
        self.encoding_window.clear()
        self.last_prediction_metrics = {}

    def _reset_state_tracking(self):
        self.face_state_window.clear()
        self.stable_face_class = "normal"

    def _smooth_face_state(self, face_class):
        self.face_state_window.append(face_class)
        counts = Counter(self.face_state_window)
        dominant_state, dominant_votes = counts.most_common(1)[0]
        if dominant_votes >= min(STATE_MIN_VOTES, len(self.face_state_window)):
            self.stable_face_class = dominant_state
        return self.stable_face_class

    def _reset_identity_progress(self, clear_locked_state=False):
        self.face_detected_at = None
        self.single_face_started_at = None
        self.recognized_name = None
        self.recognized_confidence = 0.0
        self.confirmed_match_frames = 0
        self.last_predicted_name = None
        self._reset_temporal_smoothing()
        if clear_locked_state:
            self.locked_face_class = None
            self._reset_state_tracking()
            self.previous_face_box = None

    @staticmethod
    def _is_face_moving_too_much(previous_box, current_box, frame_shape):
        if previous_box is None or current_box is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return False
        prev_top, prev_right, prev_bottom, prev_left = previous_box
        cur_top, cur_right, cur_bottom, cur_left = current_box
        prev_center_x = (prev_left + prev_right) / 2.0
        prev_center_y = (prev_top + prev_bottom) / 2.0
        cur_center_x = (cur_left + cur_right) / 2.0
        cur_center_y = (cur_top + cur_bottom) / 2.0
        shift_x = abs(cur_center_x - prev_center_x) / float(frame_width)
        shift_y = abs(cur_center_y - prev_center_y) / float(frame_height)
        return max(shift_x, shift_y) > MAX_FACE_CENTER_SHIFT_RATIO

    def _get_distance_guidance(self, face_width_ratio):
        if self.distance_guidance == "too_far":
            if face_width_ratio >= FACE_WIDTH_MIN_CLEAR_RATIO:
                self.distance_guidance = None
            else:
                return "too_far"
        elif self.distance_guidance == "too_close":
            if face_width_ratio <= FACE_WIDTH_MAX_CLEAR_RATIO:
                self.distance_guidance = None
            else:
                return "too_close"

        if face_width_ratio < MIN_FACE_WIDTH_RATIO:
            self.distance_guidance = "too_far"
            return "too_far"
        if face_width_ratio > MAX_FACE_WIDTH_RATIO:
            self.distance_guidance = "too_close"
            return "too_close"
        return None

    def _init_tracker(self, frame, box):
        top, right, bottom, left = self._clip_box(box, frame.shape)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray_frame[top:bottom, left:right]
        if template.size == 0 or template.shape[0] < 12 or template.shape[1] < 12:
            self._reset_tracker()
            return
        self.tracked_box = (top, right, bottom, left)
        self.tracker_template = template.copy()
        self.tracker_lost_frames = 0

    def _update_tracker(self, frame):
        if self.tracked_box is None or self.tracker_template is None:
            return None

        top, right, bottom, left = self.tracked_box
        face_width = right - left
        face_height = bottom - top
        if face_width <= 0 or face_height <= 0:
            self._reset_tracker()
            return None

        margin_x = int(face_width * TRACK_SEARCH_MARGIN_RATIO)
        margin_y = int(face_height * TRACK_SEARCH_MARGIN_RATIO)
        search_top, search_right, search_bottom, search_left = self._clip_box(
            (
                top - margin_y,
                right + margin_x,
                bottom + margin_y,
                left - margin_x,
            ),
            frame.shape,
        )

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_region = gray_frame[search_top:search_bottom, search_left:search_right]
        if (
            search_region.size == 0
            or search_region.shape[0] < self.tracker_template.shape[0]
            or search_region.shape[1] < self.tracker_template.shape[1]
        ):
            self._reset_tracker()
            return None

        result = cv2.matchTemplate(search_region, self.tracker_template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < TRACK_TEMPLATE_MATCH_THRESHOLD:
            self.tracker_lost_frames += 1
            if self.tracker_lost_frames >= 2:
                self._reset_tracker()
            return None

        match_left = search_left + max_loc[0]
        match_top = search_top + max_loc[1]
        match_bottom = match_top + self.tracker_template.shape[0]
        match_right = match_left + self.tracker_template.shape[1]
        self.tracked_box = self._clip_box((match_top, match_right, match_bottom, match_left), frame.shape)
        self.tracker_lost_frames = 0
        return self.tracked_box

    def _emit_console_summary(self, force=False):
        current_prediction = self.names[0] if self.names else "None"
        current_confidence = self.confidences[0] if self.confidences else 0.0
        recognized_label = self.recognized_name if self.recognized_name is not None else "Pending"
        summary = (
            "[Recognize] "
            f"detect={self.detected_frame_count} "
            f"attempts={self.recognition_attempt_count} "
            f"pred={current_prediction} "
            f"conf={current_confidence:.1f}% "
            f"confirmed={self.confirmed_match_frames} "
            f"accepted={recognized_label} "
            f"detector={self.backend_text} "
            f"status={self.status_text}"
        )
        if force or summary != self._last_console_summary:
            print(summary, flush=True)
            self._last_console_summary = summary

    def _update_online_learning(self, name, encoding):
        # Online learning: if confident, store new embedding dynamically
        if not ENABLE_ONLINE_LEARNING:
            return
        if self.added_online > 2:
            return
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self.base_encodings.append(np.asarray(encoding, dtype=np.float64)[:128])
        self.base_names.append(name)
        self.added_online += 1
        # Save state locally in background or next session
        save_model({"encodings": self.known_encodings, "names": self.known_names}, self.data_file)

    def _predict_from_full_gallery(self, encoding, backend_label, face_class="normal"):
        if not self.base_encodings:
            return {
                "name": "Unknown",
                "confidence": 0.0,
                "backend": "NoData",
            }

        face_distances = face_recognition.face_distance(self.base_encodings, encoding[:128])
        best_match_index = int(np.argmin(face_distances))
        best_distance = float(face_distances[best_match_index])
        nearest_name = self.base_names[best_match_index]
        nearest_confidence = calculate_normalized_confidence(best_distance, self.threshold)

        print(f"Predicted: {nearest_name}")
        print(f"Distance: {best_distance:.4f} (Threshold: {self.threshold})")

        confused = False
        if len(face_distances) > 1:
            sorted_indices = np.argsort(face_distances)
            for idx in sorted_indices[1:]:
                if self.base_names[idx] != nearest_name:
                    second_best_distance = float(face_distances[idx])
                    # Ratio test chong nham lan: 0.88 cho phu kien de dam bao an ninh tuyet doi
                    ratio_threshold = 0.88 if face_class != "normal" else 0.85
                    if best_distance / max(second_best_distance, 1e-6) > ratio_threshold:
                        confused = True
                        print(f"-> Tu choi do nham lan giua {nearest_name} ({best_distance:.3f}) va {self.base_names[idx]} ({second_best_distance:.3f})")
                    break
                    
        if confused:
            return {
                "name": "Unknown",
                "confidence": 0.0,
                "backend": "Confused_Margin",
            }

        # Strictly reject if distance > threshold
        if best_distance <= self.threshold:
            return {
                "name": nearest_name,
                "confidence": nearest_confidence,
                "backend": backend_label,
                "distance": best_distance,
                "distance_normalized": best_distance / max(self.threshold, 1e-6),
                "distance_scale": "base_l2",
                "distance_threshold": self.threshold,
                "encoding": encoding
            }

        return {
            "name": "Unknown",
            "confidence": 0.0,
            "backend": "Unknown",
            "distance": best_distance,
            "distance_normalized": best_distance / max(self.threshold, 1e-6),
            "distance_scale": "base_l2",
            "distance_threshold": self.threshold,
        }

    def _predict_from_combined_gallery(self, encoding, frame, face_box, face_class):
        if not self.combined_encodings or frame is None or face_box is None:
            return self._predict_from_full_gallery(
                encoding,
                backend_label=f"DeepEmbedding_L2 ({face_class})",
            )

        upper_face_encoding = get_full_upper_encoding(face_box, frame)
        if upper_face_encoding.size == 0:
            return self._predict_from_full_gallery(
                encoding,
                backend_label=f"DeepEmbedding_L2 ({face_class})",
            )

        base_query = encoding[:128]
        upper_query = upper_face_encoding

        gallery_base = np.vstack([enc[:128] for enc in self.combined_encodings])
        gallery_upper = np.vstack([enc[128:] for enc in self.combined_encodings])

        distances_base = np.linalg.norm(gallery_base - base_query, axis=1)
        distances_upper = np.linalg.norm(gallery_upper - upper_query, axis=1)

        # Trong so dong (Dynamic Weights) chong nhan dien nham
        if face_class == "mask_glasses":
            w_base = 0.70   # Deep L2 van cuc ky tot, nen tin tuong 70%
            w_upper = 0.30  # Dac diem thu cong chi dung de phu tro 30%
            upper_scale = 400.0 # Giam nhe do lon cua Vector thu cong
        else:
            w_base = 0.80   # Voi mat na thuong, tin tuong Deep L2 len toi 80%
            w_upper = 0.20
            upper_scale = 300.0

        distances = w_base * distances_base + w_upper * (distances_upper / upper_scale)
        best_match_index = int(np.argmin(distances))
        best_distance = float(distances[best_match_index])
        nearest_name = self.combined_names[best_match_index]
        threshold = self.threshold
        confidence = calculate_normalized_confidence(best_distance, threshold)

        print(f"Predicted: {nearest_name}")
        print(f"Distance: {best_distance:.4f} (Threshold: {threshold:.2f})")

        confused = False
        if len(distances) > 1:
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices[1:]:
                if self.combined_names[idx] != nearest_name:
                    second_best_distance = float(distances[idx])
                    # Tang bao mat: Nguong 0.88 de ngan chan nguoi giong nhau mo tu cua nhau
                    if best_distance / max(second_best_distance, 1e-6) > 0.88:
                        confused = True
                        print(
                            "-> Tu choi do nham lan giua "
                            f"{nearest_name} ({best_distance:.3f}) va "
                            f"{self.combined_names[idx]} ({second_best_distance:.3f})"
                        )
                    break

        if confused:
            return {
                "name": "Unknown",
                "confidence": 0.0,
                "backend": f"Confused_{face_class}",
            }

        if best_distance <= threshold:
            return {
                "name": nearest_name,
                "confidence": confidence,
                "backend": f"UpperFace_{face_class}",
                "distance": best_distance,
                "distance_normalized": best_distance / max(threshold, 1e-6),
                "distance_scale": "weighted_combined",
                "distance_threshold": threshold,
                "encoding": encoding,
            }

        return {
            "name": "Unknown",
            "confidence": 0.0,
            "backend": f"Unknown_{face_class}",
            "distance": best_distance,
            "distance_normalized": best_distance / max(threshold, 1e-6),
            "distance_scale": "weighted_combined",
            "distance_threshold": threshold,
        }

    def _predict_identity(self, encoding, frame=None, face_box=None, face_class="normal"):
        if face_class in {"mask", "mask_glasses"}:
            return self._predict_from_combined_gallery(encoding, frame, face_box, face_class)
        backend_label = "DeepEmbedding_L2"
        if face_class != "normal":
            backend_label = f"DeepEmbedding_L2 ({face_class})"
        return self._predict_from_full_gallery(encoding, backend_label, face_class)

    @staticmethod
    def _is_soft_face_class_transition(previous_face_class, current_face_class):
        classes = {previous_face_class, current_face_class}
        if classes <= {"normal", "glasses"}:
            return True
        if classes <= {"mask", "mask_glasses"}:
            return True
        if classes <= {"glasses", "mask_glasses"}:
            return True
        return False

    @staticmethod
    def _format_optional_metric(value, digits=3):
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    def process_frame(self, frame):
        display_frame = frame.copy()
        guide_rect = get_guide_rect(display_frame.shape)
        guide_left, guide_top, guide_right, guide_bottom = guide_rect
        is_aligned = False

        small_frame = cv2.resize(
            display_frame,
            (0, 0),
            fx=FRAME_PROCESS_SCALE,
            fy=FRAME_PROCESS_SCALE,
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if self.process_this_frame:
            current_time = time.time()
            detector_backend = "Khoi tao"
            face_class = "normal"

            def scale_boxes(boxes):
                scale_back = 1.0 / FRAME_PROCESS_SCALE
                return [
                    (
                        int(top * scale_back),
                        int(right * scale_back),
                        int(bottom * scale_back),
                        int(left * scale_back),
                    )
                    for top, right, bottom, left in boxes
                ]

            def boxes_to_small(boxes):
                return [self._scale_box_to_small(box) for box in boxes]

            # Pipeline Stage 1: Track first, detect only when tracking is missing or stale.
            tracked_box = None
            should_refresh_detection = current_time - self.last_detection_time >= TRACK_REFRESH_SECONDS
            if not should_refresh_detection:
                tracked_box = self._update_tracker(display_frame)

            if tracked_box is not None:
                scaled_boxes = [tracked_box]
                raw_boxes = boxes_to_small(scaled_boxes)
                detector_backend = "TRACK"
            else:
                raw_boxes, detector_backend = detect_face_boxes(rgb_small_frame)
                scaled_boxes = scale_boxes(raw_boxes)
                self.last_detection_time = current_time
                if len(scaled_boxes) == 1:
                    self._init_tracker(display_frame, scaled_boxes[0])
                else:
                    self._reset_tracker()

            if len(scaled_boxes) == 1:
                raw_face_class = classify_face_state(display_frame, scaled_boxes[0], self.state_svm_model)
                face_class = self._smooth_face_state(raw_face_class)
                if face_class != "normal":
                    detector_backend = f"{detector_backend} ({face_class})"

            self.boxes = scaled_boxes
            self.names = []
            self.confidences = []

            if len(self.boxes) == 0:
                self.distance_guidance = None
                self._reset_tracker()
                self._reset_identity_progress(clear_locked_state=True)
                self.backend_text = detector_backend
                self.show_identity_label = False
                self.status_text = "Chua thay khuon mat"
                self.status_color = (0, 0, 255)
            elif len(self.boxes) > 1:
                self.distance_guidance = None
                self._reset_tracker()
                self.detected_frame_count += len(self.boxes)
                self._reset_identity_progress(clear_locked_state=True)
                self.multi_face_blocked_until = current_time + MULTI_FACE_BLOCK_SECONDS
                self.backend_text = f"{detector_backend}: Multi-face blocked"
                self.show_identity_label = False
                self.status_text = "Chi de 1 khuon mat trong khung"
                self.status_color = (0, 0, 255)
            else:
                self.detected_frame_count += 1
                face_box = self.boxes[0]
                brightness, sharpness = measure_face_quality(display_frame, face_box)
                is_aligned = is_face_inside_guide(face_box, guide_rect)
                face_width_ratio = get_face_width_ratio(display_frame.shape, face_box)
                distance_guidance = self._get_distance_guidance(face_width_ratio)
                is_frontal, face_pose_metrics = is_face_frontal_enough(display_frame, face_box)
                moving_too_much = self._is_face_moving_too_much(
                    self.previous_face_box,
                    face_box,
                    display_frame.shape,
                )
                self.previous_face_box = face_box
                raw_face_class_for_metrics = face_class
                if self.mode == 'keep' and not is_frontal:
                    face_class = "normal"
                self.last_quality_metrics = {
                    "brightness": float(brightness),
                    "sharpness": float(sharpness),
                    "face_width_ratio": float(face_width_ratio),
                    "aligned": bool(is_aligned),
                    "frontal": bool(is_frontal),
                    "face_pose": dict(face_pose_metrics),
                    "moving_too_much": bool(moving_too_much),
                    "face_class": face_class,
                    "raw_face_class": raw_face_class_for_metrics,
                }

                if self.locked_face_class is None:
                    self.locked_face_class = face_class
                elif face_class != self.locked_face_class:
                    previous_face_class = self.locked_face_class
                    if self.mode == 'take' and self._is_soft_face_class_transition(previous_face_class, face_class):
                        self.locked_face_class = face_class
                        detector_backend = f"{detector_backend}: Soft state {previous_face_class}->{face_class}"
                    else:
                        self._reset_identity_progress(clear_locked_state=True)
                        self.locked_face_class = face_class
                        self.backend_text = f"{detector_backend}: State reset {previous_face_class}->{face_class}"
                        self.show_identity_label = False
                        self.names = []
                        self.confidences = []
                        self.status_text = f"Trang thai thay doi ({previous_face_class} -> {face_class}), vui long giu yen lai"
                        self.status_color = (0, 255, 255)
                        self.process_this_frame = not self.process_this_frame
                        curr_time = time.time()
                        fps = 1 / (curr_time - self.prev_time) if curr_time > self.prev_time else 0
                        self.prev_time = curr_time
                        apply_focus_overlay(display_frame, guide_rect, is_aligned=is_aligned)
                        info_lines = [
                            f"Detect ok: {self.detected_frame_count}",
                            f"So lan match: {self.recognition_attempt_count}",
                            f"Du doan: None",
                            "Tin cay: 0.0%",
                            f"Da xac nhan: {self.confirmed_match_frames}",
                            f"Detector: {self.backend_text}",
                        ]
                        self.summary_text = " | ".join(info_lines)
                        self._emit_console_summary()
                        return {
                            "frame": display_frame,
                            "fps": fps,
                            "status_text": self.status_text,
                            "status_color": self.status_color,
                            "backend_text": self.backend_text,
                            "summary_text": self.summary_text,
                            "diagnostics": {
                                "quality": dict(self.last_quality_metrics),
                                "prediction": dict(self.last_prediction_metrics),
                            },
                            "done": self.done,
                            "result": self.result,
                        }
                
                # Pipeline Stage 3: Business Logic - HARD BLOCK
                if self.mode == 'keep' and not is_frontal:
                    self.status_text = "Hay nhin thang vao camera, khong nghieng mat"
                    self.status_color = (0, 255, 255)
                    self._reset_identity_progress(clear_locked_state=False)
                    self.backend_text = f"{detector_backend}: Pose"
                    self.show_identity_label = False
                    self.names = []
                    self.confidences = []
                elif self.mode == 'keep' and face_class != "normal":
                    self.status_text = f"Tu choi ({face_class}). Vui long de mat tran."
                    self.status_color = (0, 0, 255)
                    self._reset_identity_progress(clear_locked_state=True)
                    self.names = []
                    self.confidences = []
                elif current_time < self.multi_face_blocked_until:
                    remaining = self.multi_face_blocked_until - current_time
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = f"{detector_backend}: Multi-face cooldown"
                    self.show_identity_label = False
                    self.status_text = f"Chi de 1 khuon mat on dinh {remaining:.1f}s"
                    self.status_color = (0, 0, 255)
                elif not is_aligned:
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = detector_backend
                    self.show_identity_label = False
                    self.status_text = "Hay nhin thang vao camera va can giua khuon mat vao khung"
                    self.status_color = (0, 255, 255)
                elif moving_too_much:
                    self._reset_identity_progress(clear_locked_state=False)
                    self.backend_text = f"{detector_backend}: Motion"
                    self.show_identity_label = False
                    self.status_text = "Hay nhin thang vao camera va giu yen de nhan dien tot hon"
                    self.status_color = (0, 255, 255)
                elif distance_guidance == "too_far":
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = f"{detector_backend}: Distance"
                    self.show_identity_label = False
                    self.status_text = "Tien gan hon mot chut"
                    self.status_color = (0, 255, 255)
                elif distance_guidance == "too_close":
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = f"{detector_backend}: Distance"
                    self.show_identity_label = False
                    self.status_text = "Lui ra mot chut"
                    self.status_color = (0, 255, 255)
                elif brightness < MIN_FACE_BRIGHTNESS:
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = f"{detector_backend}: Low light"
                    self.show_identity_label = False
                    self.status_text = "Anh qua toi, hay tang anh sang"
                    self.status_color = (0, 0, 255)
                elif sharpness < MIN_FACE_SHARPNESS:
                    self._reset_identity_progress(clear_locked_state=True)
                    self.backend_text = f"{detector_backend}: Blur"
                    self.show_identity_label = False
                    self.status_text = "Hay giu yen de camera lay net"
                    self.status_color = (0, 255, 255)
                else:
                    allow_identity_check = True

                    required_consistent_frames = REQUIRED_CONSISTENT_FRAMES
                    if face_class in {"mask", "mask_glasses"}:
                        required_consistent_frames = MASK_REQUIRED_CONSISTENT_FRAMES
                    if self.single_face_started_at is None:
                        self.single_face_started_at = current_time

                    stable_seconds = current_time - self.single_face_started_at
                    if stable_seconds < SINGLE_FACE_STABLE_SECONDS:
                        self.face_detected_at = None
                        self.recognized_name = None
                        self.recognized_confidence = 0.0
                        self.backend_text = f"{detector_backend}: Single-face check"
                        self.show_identity_label = False
                        self.status_text = f"Xac nhan 1 khuon mat {SINGLE_FACE_STABLE_SECONDS - stable_seconds:.1f}s"
                        self.status_color = (0, 255, 255)
                        self.names = []
                        self.confidences = []
                    else:
                        allow_identity_check = True

                    if allow_identity_check:
                        if self.face_detected_at is None:
                            self.face_detected_at = current_time

                        # Pipeline Stage 3: Recognition (Delayed encoding for performance)
                        enhanced_rgb_small = enhance_lighting(rgb_small_frame)
                        frame_encodings = face_recognition.face_encodings(enhanced_rgb_small, raw_boxes)

                        if not self.known_encodings or not frame_encodings:
                            self._reset_temporal_smoothing()
                            self.last_predicted_name = None
                            self.confirmed_match_frames = 0
                            self.recognized_name = None
                            self.recognized_confidence = 0.0
                            self.names.append("Unknown")
                            self.confidences.append(0.0)
                            self.backend_text = "NoData" if not self.known_encodings else "ExtractError"
                        else:
                            self.recognition_attempt_count += 1
                            self.encoding_window.append(frame_encodings[0])
                            avg_encoding = np.mean(self.encoding_window, axis=0)

                            prediction = self._predict_identity(
                                avg_encoding, 
                                frame=display_frame,
                                face_box=face_box,
                                face_class=face_class,
                            )
                            name = prediction["name"]
                            confidence = prediction["confidence"]
                            self.backend_text = prediction.get("backend", "Unknown")
                            self.last_prediction_metrics = {
                                "distance": prediction.get("distance"),
                                "distance_normalized": prediction.get("distance_normalized"),
                                "distance_scale": prediction.get("distance_scale", ""),
                                "distance_threshold": prediction.get("distance_threshold"),
                                "face_class": face_class,
                                "backend": self.backend_text,
                            }

                            if name != "Unknown":
                                self.prediction_window.append((name, confidence))
                                vote_counts = Counter(item[0] for item in self.prediction_window)
                                voted_name, voted_frames = vote_counts.most_common(1)[0]
                                voted_confidences = [
                                    item_confidence
                                    for item_name, item_confidence in self.prediction_window
                                    if item_name == voted_name
                                ]
                                self.last_predicted_name = voted_name
                                self.confirmed_match_frames = voted_frames
                                if voted_frames >= TEMPORAL_MIN_VOTES:
                                    self.recognized_name = voted_name
                                    self.recognized_confidence = float(np.mean(voted_confidences))
                                    
                                    # Trigger Online Learning update
                                    distance = prediction.get("distance", 1.0)
                                    distance_scale = prediction.get("distance_scale")
                                    if (
                                        name == voted_name
                                        and distance_scale == "base_l2"
                                        and distance < 0.35
                                    ):
                                        self._update_online_learning(voted_name, prediction["encoding"])
                                else:
                                    self.recognized_name = None
                                    self.recognized_confidence = 0.0
                            else:
                                self._reset_temporal_smoothing()
                                self.last_predicted_name = None
                                self.confirmed_match_frames = 0
                                self.recognized_name = None
                                self.recognized_confidence = 0.0

                            self.names.append(name)
                            self.confidences.append(confidence)

                        elapsed_seconds = current_time - self.face_detected_at
                        remaining_seconds = max(0.0, self.min_display_seconds - elapsed_seconds)

                        if remaining_seconds > 0:
                            self.status_text = f"Giu yen {remaining_seconds:.1f}s"
                            self.status_color = (0, 255, 255)
                            self.show_identity_label = False
                        elif self.recognized_name is not None:
                            self.status_text = f"Xac thuc thanh cong: {self.recognized_name}"
                            self.status_color = (0, 255, 0)
                            self.show_identity_label = True
                        elif self.last_predicted_name is not None and self.confirmed_match_frames > 0:
                            remaining_frames = required_consistent_frames - self.confirmed_match_frames
                            self.status_text = f"Xac nhan them {remaining_frames} khung hinh"
                            self.status_color = (0, 255, 255)
                            self.show_identity_label = False
                        else:
                            self.status_text = "Khong xac thuc duoc. Thu lai"
                            self.status_color = (0, 0, 255)
                            self.show_identity_label = False

                        if (
                            self.recognized_name is not None
                            and elapsed_seconds >= self.min_display_seconds
                        ):
                            result_diagnostics = {
                                "quality": dict(self.last_quality_metrics),
                                "prediction": dict(self.last_prediction_metrics),
                            }
                            self.done = True
                            self.result = {
                                "name": self.recognized_name,
                                "confidence": self.recognized_confidence,
                                "has_known_faces": bool(self.known_encodings),
                                "backend": self.backend_text,
                                "diagnostics": result_diagnostics,
                            }
                        elif (
                            self.face_detected_at is not None
                            and self.timeout_seconds is not None
                            and elapsed_seconds >= self.timeout_seconds
                        ):
                            result_diagnostics = {
                                "quality": dict(self.last_quality_metrics),
                                "prediction": dict(self.last_prediction_metrics),
                            }
                            self.done = True
                            self.result = {
                                "name": "Unknown",
                                "confidence": 0.0,
                                "has_known_faces": bool(self.known_encodings),
                                "backend": self.backend_text,
                                "diagnostics": result_diagnostics,
                            }

        self.process_this_frame = not self.process_this_frame

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if curr_time > self.prev_time else 0
        self.prev_time = curr_time

        apply_focus_overlay(display_frame, guide_rect, is_aligned=is_aligned)
        info_lines = [
            f"Detect ok: {self.detected_frame_count}",
            f"So lan match: {self.recognition_attempt_count}",
            f"Du doan: {self.names[0] if self.names else 'None'}",
            f"Tin cay: {self.confidences[0]:.1f}%" if self.confidences else "Tin cay: 0.0%",
            f"Da xac nhan: {self.confirmed_match_frames}",
            f"Detector: {self.backend_text}",
        ]
        self.summary_text = " | ".join(info_lines)
        self._emit_console_summary(force=self.done)
        for index, (top, right, bottom, left) in enumerate(self.boxes):
            name = self.names[index] if index < len(self.names) else ""
            confidence = self.confidences[index] if index < len(self.confidences) else 0.0
            box_color = (0, 255, 0) if self.recognized_name is not None else (0, 165, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
            if self.show_identity_label and name:
                label = f"{name}: {confidence:.1f}%"
                cv2.putText(
                    display_frame,
                    label,
                    (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2,
                )

        return {
            "frame": display_frame,
            "fps": fps,
            "status_text": self.status_text,
            "status_color": self.status_color,
            "backend_text": self.backend_text,
            "summary_text": self.summary_text,
            "diagnostics": {
                "quality": dict(self.last_quality_metrics),
                "prediction": dict(self.last_prediction_metrics),
            },
            "done": self.done,
            "result": self.result,
        }


def recognize_face_live(
    data_file="encodings.pickle",
    camera_index=0,
    threshold=0.42,
    timeout_seconds=10,
    min_display_seconds=0,
    mode='take',
    window_width=None,
    window_height=None,
    window_title="Face Recognition",
    close_window=True,
):
    session = FaceRecognitionSession(
        data_file=data_file,
        threshold=threshold,
        timeout_seconds=timeout_seconds,
        min_display_seconds=min_display_seconds,
        mode=mode,
    )
    cap = create_camera(
        camera_index=camera_index,
        width=CAMERA_FRAME_WIDTH,
        height=CAMERA_FRAME_HEIGHT,
    )
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    cv2.namedWindow(window_title)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read frame from webcam.")

            packet = session.process_frame(frame)
            display_frame = packet["frame"].copy()
            cv2.rectangle(display_frame, (0, 0), (360, 90), (0, 0, 0), -1)
            cv2.putText(display_frame, f"FPS: {int(packet['fps'])}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, packet["status_text"], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, packet["status_color"], 2)
            cv2.putText(display_frame, "Nhan q de dung", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.imshow(window_title, display_frame)

            if packet["done"]:
                return packet["result"]

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if close_window:
            cv2.destroyWindow(window_title)

    return {
        "name": "Unknown",
        "confidence": 0.0,
        "has_known_faces": bool(session.known_encodings),
        "backend": session.backend_text,
    }


def main():
    result = recognize_face_live()
    print(f"Result: {result['name']} ({result['confidence']:.1f}%)")


if __name__ == "__main__":
    main()
