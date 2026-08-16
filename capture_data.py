import json
import math
import os
import time
from collections import Counter, deque

import cv2
import face_recognition
import numpy as np

from camera_utils import create_camera
from svm_model import load_state_svm, predict_face_state_svm_details

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
MIN_FACE_BRIGHTNESS = 50.0
MIN_FACE_SHARPNESS = 20.0
MIN_FACE_WIDTH_RATIO = 0.16
MAX_FACE_WIDTH_RATIO = 0.50
ACCESSORY_BLOCK_SECONDS = 0.5
NORMAL_STABLE_SECONDS_BEFORE_CAPTURE = 1.2
NORMAL_CAPTURE_REQUIRED_CONSECUTIVE_FRAMES = 8
LOWER_FACE_OCCLUSION_RATIO = 0.42
EYE_GLARE_BRIGHT_RATIO = 0.10
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

_HAAR_CASCADE = None
_YUNET_DETECTOR = None


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
    # The Tkinter/PIL camera view draws the modern scan frame; keep capture frames clean.
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
                
    max_accessory_score = max(mask_score, glasses_score)
    if (
        normal_prob >= STATE_NORMAL_CLEAR_PROBABILITY
        and max_accessory_score < STATE_ACCESSORY_SUSPECT_SCORE
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
                return boxes, backend["model"].upper()

    if last_error is not None:
        return [], "Detect unavailable"
    return [], "No face"


class FaceCaptureSession:
    def __init__(
        self,
        user_name,
        dataset_dir="dataset",
        max_images=12,
        capture_interval=0.25,
        capture_mode="normal",
        file_tag=None,
    ):
        self.user_name = user_name.strip()
        if not self.user_name:
            raise ValueError("User name cannot be empty.")

        self.dataset_dir = dataset_dir
        self.max_images = max_images
        self.capture_interval = capture_interval
        self.capture_mode = capture_mode
        self.allow_mask = capture_mode in {"mask", "mask_glasses"}
        self.allow_glasses = capture_mode in {"glasses", "mask_glasses"}
        default_tag_map = {
            "normal": "nomask",
            "mask": "mask",
            "glasses": "glasses",
            "mask_glasses": "mask_glasses",
        }
        self.file_tag = (file_tag or default_tag_map.get(capture_mode, "sample")).strip("_")
        self.user_dir = os.path.join(dataset_dir, self.user_name)
        os.makedirs(self.user_dir, exist_ok=True)

        self.count = 0
        self.detected_frame_count = 0
        self.valid_frame_count = 0
        self.last_capture_time = 0.0
        self.done = False
        self.backend_text = "Khoi tao"
        self.summary_text = ""
        self._last_console_summary = None
        self.accessory_blocked_until = 0.0
        self.normal_ready_started_at = None
        self.normal_candidate_frames = 0
        self.face_state_window = deque(maxlen=STATE_WINDOW_SIZE)
        self.stable_face_class = "normal"
        if capture_mode == "mask_glasses":
            self.status_text = "Deo khau trang va kinh, dua khuon mat vao khung de bat dau chup"
        elif self.allow_mask:
            self.status_text = "Deo khau trang va dua khuon mat vao khung de bat dau chup"
        elif self.allow_glasses:
            self.status_text = "Deo kinh va dua khuon mat vao khung de bat dau chup"
        else:
            self.status_text = "Dua khuon mat vao khung de bat dau chup"
        self.status_color = (0, 255, 255)
        self.state_svm_model = load_state_svm()

    def _emit_console_summary(self, force=False):
        summary = (
            f"[Capture:{self.user_name}/{self.capture_mode}] "
            f"detect={self.detected_frame_count} "
            f"valid={self.valid_frame_count} "
            f"saved={self.count}/{self.max_images} "
            f"detector={self.backend_text} "
            f"status={self.status_text}"
        )
        if force or summary != self._last_console_summary:
            print(summary, flush=True)
            self._last_console_summary = summary

    def _reset_state_tracking(self):
        self.face_state_window.clear()
        self.stable_face_class = "normal"
        self.normal_candidate_frames = 0

    def _reset_normal_capture_gate(self):
        self.normal_ready_started_at = None
        self.normal_candidate_frames = 0

    def _smooth_face_state(self, face_class):
        self.face_state_window.append(face_class)
        counts = Counter(self.face_state_window)
        dominant_state, dominant_votes = counts.most_common(1)[0]
        if dominant_votes >= min(STATE_MIN_VOTES, len(self.face_state_window)):
            self.stable_face_class = dominant_state
        return self.stable_face_class

    def process_frame(self, frame):
        display_frame = frame.copy()
        guide_rect = get_guide_rect(display_frame.shape)
        is_aligned = False
        small_frame = cv2.resize(display_frame, (0, 0), fx=FRAME_PROCESS_SCALE, fy=FRAME_PROCESS_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        prefer_yunet = self.allow_mask or self.allow_glasses
        boxes, self.backend_text = detect_face_boxes(rgb_small, prefer_yunet=prefer_yunet)

        if len(boxes) == 1:
            self.detected_frame_count += 1
            scale_back = 1.0 / FRAME_PROCESS_SCALE
            scaled_box = tuple(int(round(value * scale_back)) for value in boxes[0])
            brightness, sharpness = measure_face_quality(frame, scaled_box)
            raw_face_class = classify_face_state(frame, scaled_box, self.state_svm_model)
            face_class = self._smooth_face_state(raw_face_class)
            is_aligned = is_face_inside_guide(scaled_box, guide_rect)
            face_width_ratio = get_face_width_ratio(display_frame.shape, scaled_box)
            is_frontal, face_pose_metrics = is_face_frontal_enough(frame, scaled_box)
            current_time = time.time()
            normal_capture_state = raw_face_class == "normal" and face_class == "normal"
            print(
                "Debug Quality: "
                f"State={face_class}, Raw={raw_face_class}, Bright={brightness:.1f}, "
                f"Sharp={sharpness:.1f}, Ratio={face_width_ratio:.2f}, "
                f"Frontal={is_frontal}"
            )
            
            # HARD BLOCK: Stop processing if face is not normal
            capture_state_ok = (
                True if (self.allow_mask or self.allow_glasses) else normal_capture_state
            )
            if not is_frontal and not self.allow_mask and not self.allow_glasses:
                self._reset_normal_capture_gate()
                self.status_text = "Hay nhin thang vao camera, khong nghieng mat"
                self.status_color = (0, 255, 255)
            elif not normal_capture_state and not self.allow_mask and not self.allow_glasses:
                self.accessory_blocked_until = current_time + ACCESSORY_BLOCK_SECONDS
                self._reset_normal_capture_gate()
                rejected_state = face_class if face_class != "normal" else raw_face_class
                reject_label = {
                    "glasses": "deo kinh",
                    "mask": "deo khau trang",
                    "mask_glasses": "deo khau trang + kinh",
                }.get(rejected_state, rejected_state)
                self.status_text = f"Tu choi ({reject_label}). Vui long de mat tran."
                self.status_color = (0, 0, 255)
                # Luon giu boxes de hien thi tren UI, chi chan viec chup anh o cac khoi elif/else duoi
            elif (
                not self.allow_mask
                and not self.allow_glasses
                and current_time < self.accessory_blocked_until
            ):
                self._reset_normal_capture_gate()
                remaining = self.accessory_blocked_until - current_time
                self.status_text = f"Dang khoa chup {remaining:.1f}s de kiem tra lai mat tran"
                self.status_color = (0, 0, 255)
            elif not is_aligned:
                self._reset_normal_capture_gate()
                self.status_text = "Can giua khuon mat vao khung mau vang"
                self.status_color = (0, 255, 255)
            elif face_width_ratio < MIN_FACE_WIDTH_RATIO:
                self._reset_normal_capture_gate()
                self.status_text = "Tien gan hon mot chut"
                self.status_color = (0, 255, 255)
            elif face_width_ratio > MAX_FACE_WIDTH_RATIO:
                self._reset_normal_capture_gate()
                self.status_text = "Lui ra mot chut"
                self.status_color = (0, 255, 255)
            elif brightness < MIN_FACE_BRIGHTNESS:
                self._reset_normal_capture_gate()
                self.status_text = "Anh qua toi, hay tang anh sang"
                self.status_color = (0, 0, 255)
            elif sharpness < MIN_FACE_SHARPNESS:
                self._reset_normal_capture_gate()
                self.status_text = "Hay giu yen de anh ro hon"
                self.status_color = (0, 255, 255)
            elif not self.allow_mask and not self.allow_glasses:
                if self.normal_ready_started_at is None:
                    self.normal_ready_started_at = current_time
                    self.normal_candidate_frames = 0
                self.normal_candidate_frames += 1
                stable_seconds = current_time - self.normal_ready_started_at
                if (
                    stable_seconds < NORMAL_STABLE_SECONDS_BEFORE_CAPTURE
                    or self.normal_candidate_frames < NORMAL_CAPTURE_REQUIRED_CONSECUTIVE_FRAMES
                ):
                    self.status_text = (
                        "Xac nhan mat tran "
                        f"{max(0.0, NORMAL_STABLE_SECONDS_BEFORE_CAPTURE - stable_seconds):.1f}s"
                    )
                    self.status_color = (0, 255, 255)
                elif current_time - self.last_capture_time >= self.capture_interval:
                    self.count += 1
                    image_path = os.path.join(
                        self.user_dir,
                        f"{self.user_name}_{self.file_tag}_{self.count}.jpg",
                    )
                    cv2.imwrite(image_path, frame)
                    metadata_path = os.path.splitext(image_path)[0] + ".json"
                    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                        json.dump(
                            {
                                "image_file": os.path.basename(image_path),
                                "image_shape": list(frame.shape[:2]),
                                "face_box": list(scaled_box),
                                "detector": self.backend_text,
                                "capture_mode": self.capture_mode,
                                "face_pose": face_pose_metrics,
                            },
                            metadata_file,
                        )
                    self.last_capture_time = current_time
            elif current_time - self.last_capture_time >= self.capture_interval:
                self.count += 1
                image_path = os.path.join(
                    self.user_dir,
                    f"{self.user_name}_{self.file_tag}_{self.count}.jpg",
                )
                cv2.imwrite(image_path, frame)
                metadata_path = os.path.splitext(image_path)[0] + ".json"
                with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                    json.dump(
                        {
                            "image_file": os.path.basename(image_path),
                            "image_shape": list(frame.shape[:2]),
                            "face_box": list(scaled_box),
                            "detector": self.backend_text,
                            "capture_mode": self.capture_mode,
                            "face_pose": face_pose_metrics,
                        },
                        metadata_file,
                    )
                self.last_capture_time = current_time

            for (top, right, bottom, left) in boxes:
                if not is_frontal and not self.allow_mask and not self.allow_glasses:
                    draw_color = (0, 255, 255)
                else:
                    draw_color = (0, 255, 0) if capture_state_ok else (0, 0, 255)
                top = int(round(top * scale_back))
                right = int(round(right * scale_back))
                bottom = int(round(bottom * scale_back))
                left = int(round(left * scale_back))
                cv2.rectangle(display_frame, (left, top), (right, bottom), draw_color, 2)

            progress_width = int((self.count / self.max_images) * display_frame.shape[1])
            cv2.rectangle(
                display_frame,
                (0, display_frame.shape[0] - 12),
                (progress_width, display_frame.shape[0]),
                (0, 180, 0),
                -1,
            )
            if (
                is_aligned
                and is_frontal
                and capture_state_ok
                and MIN_FACE_WIDTH_RATIO <= face_width_ratio <= MAX_FACE_WIDTH_RATIO
                and brightness >= MIN_FACE_BRIGHTNESS
                and sharpness >= MIN_FACE_SHARPNESS
                and (
                    self.allow_mask
                    or self.allow_glasses
                    or (
                        current_time >= self.accessory_blocked_until
                        and self.normal_ready_started_at is not None
                        and current_time - self.normal_ready_started_at >= NORMAL_STABLE_SECONDS_BEFORE_CAPTURE
                        and self.normal_candidate_frames >= NORMAL_CAPTURE_REQUIRED_CONSECUTIVE_FRAMES
                    )
                )
            ):
                self.valid_frame_count += 1
                if self.capture_mode == "mask_glasses":
                    capture_mode_text = "khau trang + kinh"
                elif self.allow_mask:
                    capture_mode_text = "co khau trang"
                elif self.allow_glasses:
                    capture_mode_text = "deo kinh"
                else:
                    capture_mode_text = "khong khau trang"
                self.status_text = f"Dang chup du lieu {capture_mode_text}: {self.count}/{self.max_images}"
                self.status_color = (0, 255, 0)
        elif len(boxes) > 1:
            self._reset_state_tracking()
            self.status_text = "Chi de 1 khuon mat trong khung khi dang ky"
            self.status_color = (0, 0, 255)
        else:
            self._reset_state_tracking()
            self.status_text = "Chua thay khuon mat"
            self.status_color = (0, 0, 255)

        if self.count >= self.max_images:
            self.done = True

        mode_label = {
            "normal": "Thuong",
            "mask": "Khau trang",
            "glasses": "Deo kinh",
            "mask_glasses": "Khau trang + Kinh",
        }.get(self.capture_mode, self.capture_mode)
        info_lines = [
            f"Mode: {mode_label}",
            f"Detect ok: {self.detected_frame_count}",
            f"Khung hop le: {self.valid_frame_count}",
            f"Da luu/train: {self.count}/{self.max_images}",
            f"Detector: {self.backend_text}",
        ]
        self.summary_text = " | ".join(info_lines)
        self._emit_console_summary(force=self.done)

        apply_focus_overlay(display_frame, guide_rect, is_aligned=is_aligned)

        return {
            "frame": display_frame,
            "status_text": self.status_text,
            "status_color": self.status_color,
            "backend_text": self.backend_text,
            "summary_text": self.summary_text,
            "done": self.done,
            "result": self.count,
        }


def capture_user_data(
    user_name,
    dataset_dir="dataset",
    max_images=20,
    camera_index=0,
    capture_interval=0.15,
    capture_mode="normal",
    file_tag=None,
    window_width=None,
    window_height=None,
    window_title="Face Registration",
    close_window=True,
):
    session = FaceCaptureSession(
        user_name=user_name,
        dataset_dir=dataset_dir,
        max_images=max_images,
        capture_interval=capture_interval,
        capture_mode=capture_mode,
        file_tag=file_tag,
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
        while not session.done:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read frame from webcam.")

            packet = session.process_frame(frame)
            display_frame = packet["frame"].copy()
            cv2.putText(
                display_frame,
                packet["status_text"],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                packet["status_color"],
                2,
            )
            cv2.imshow(window_title, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if close_window:
            cv2.destroyWindow(window_title)

    return session.count


def main():
    user_name = input("Nhap ten nguoi dung hoac ID: ").strip()
    saved_images = capture_user_data(user_name=user_name)
    print(f"[DONE] Total captured images: {saved_images}")


if __name__ == "__main__":
    main()
