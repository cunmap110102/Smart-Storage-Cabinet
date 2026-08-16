import os
import platform
import time

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


def _get_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_CAMERA_FRAME_WIDTH = _get_int_env("CAMERA_FRAME_WIDTH", 640)
DEFAULT_CAMERA_FRAME_HEIGHT = _get_int_env("CAMERA_FRAME_HEIGHT", 480)
DEFAULT_CAMERA_INDEX = _get_int_env("CAMERA_INDEX", 0)
DEFAULT_CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "opencv").strip().lower()
DEFAULT_CAMERA_ROTATION = _get_int_env("CAMERA_ROTATION", 0)
DEFAULT_CAMERA_FLIP = os.getenv("CAMERA_FLIP", "none").strip().lower()
DEFAULT_CAMERA_COLOR_ORDER = os.getenv("CAMERA_COLOR_ORDER", "auto").strip().lower()
DEFAULT_PI_CAMERA_FRAME_RATE = _get_float_env("PI_CAMERA_FRAME_RATE", 15.0)
DEFAULT_PI_CAMERA_AUTO_WHITE_BALANCE = _get_bool_env("PI_CAMERA_AWB_ENABLE", True)
DEFAULT_PI_CAMERA_AUTO_EXPOSURE = _get_bool_env("PI_CAMERA_AUTO_EXPOSURE", True)


def _normalize_rotation(rotation):
    if rotation in (0, 90, 180, 270):
        return rotation
    return 0


def _apply_frame_transform(frame, rotation=0, flip="none"):
    if frame is None:
        return None

    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip == "horizontal":
        frame = cv2.flip(frame, 1)
    elif flip == "vertical":
        frame = cv2.flip(frame, 0)
    elif flip == "both":
        frame = cv2.flip(frame, -1)

    return frame


def _normalize_color_order(color_order):
    if color_order in {"rgb", "bgr", "auto"}:
        return color_order
    return "auto"


def _convert_to_bgr(frame, color_order):
    if frame is None:
        return None

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.shape[2] == 4:
        if color_order == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if color_order == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def has_display():
    if os.name == "nt":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def is_raspberry_pi():
    if platform.system() != "Linux":
        return False

    model_path = "/proc/device-tree/model"
    try:
        with open(model_path, "r", encoding="utf-8") as file_obj:
            return "raspberry pi" in file_obj.read().strip().lower()
    except OSError:
        machine = platform.machine().lower()
        return machine.startswith(("arm", "aarch64"))


class OpenCVCamera:
    def __init__(self, capture, rotation=0, flip="none"):
        self.capture = capture
        self.rotation = _normalize_rotation(rotation)
        self.flip = flip

    def isOpened(self):
        return self.capture.isOpened()

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        return True, _apply_frame_transform(frame, self.rotation, self.flip)

    def release(self):
        self.capture.release()

    def set(self, prop_id, value):
        return self.capture.set(prop_id, value)


class Picamera2Camera:
    def __init__(self, width, height, rotation=0, flip="none", color_order="auto"):
        if Picamera2 is None:
            raise RuntimeError(
                "Picamera2 is not installed. Install python3-picamera2 or set CAMERA_SOURCE=opencv."
            )

        self.rotation = _normalize_rotation(rotation)
        self.flip = flip
        self.color_order = _normalize_color_order(color_order)
        self.camera = Picamera2()
        self._opened = False
        try:
            configuration = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.camera.configure(configuration)
            self._apply_camera_controls()
            self.camera.start()
            self._opened = True
            time.sleep(0.2)
        except Exception as exc:
            self.camera.close()
            raise RuntimeError(f"Could not initialize Pi Camera: {exc}") from exc

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        frame = self.camera.capture_array()
        if frame is None:
            return False, None

        frame = _convert_to_bgr(frame, self._resolve_color_order(frame))

        return True, _apply_frame_transform(frame, self.rotation, self.flip)

    def release(self):
        if not self._opened:
            return
        self.camera.stop()
        self.camera.close()
        self._opened = False

    def set(self, prop_id, value):
        return False

    def _apply_camera_controls(self):
        controls = {"FrameRate": DEFAULT_PI_CAMERA_FRAME_RATE}
        if DEFAULT_PI_CAMERA_AUTO_EXPOSURE:
            controls["ExposureTime"] = 0
        if DEFAULT_PI_CAMERA_AUTO_WHITE_BALANCE:
            controls["AwbEnable"] = True

        try:
            self.camera.set_controls(controls)
        except Exception:
            # Some camera/driver combinations expose fewer controls.
            pass

    def _resolve_color_order(self, frame):
        if self.color_order != "auto":
            return self.color_order

        # Picamera2 with RGB888 returns RGB frames, while OpenCV code downstream
        # expects BGR before converting for display/face_recognition.
        if frame is not None and frame.ndim == 3 and frame.shape[2] in {3, 4}:
            return "rgb"

        return "bgr"


def _open_opencv_camera(camera_index, width, height, rotation, flip):
    backend_candidates = [cv2.CAP_ANY]
    if platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
        backend_candidates.insert(0, cv2.CAP_V4L2)

    camera_candidates = [camera_index]
    for candidate in range(0, 6):
        if candidate not in camera_candidates:
            camera_candidates.append(candidate)

    last_error = None
    for candidate_index in camera_candidates:
        for backend in backend_candidates:
            try:
                capture = cv2.VideoCapture(candidate_index, backend)
            except TypeError:
                capture = cv2.VideoCapture(candidate_index)

            if not capture.isOpened():
                capture.release()
                last_error = (
                    f"Cannot open camera index {candidate_index} with OpenCV backend {backend}."
                )
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = capture.read()
            if ok and frame is not None:
                return OpenCVCamera(capture, rotation=rotation, flip=flip)

            capture.release()
            last_error = (
                f"Camera index {candidate_index} opened but did not return frames."
            )

    raise RuntimeError(last_error or "Cannot open camera with OpenCV.")


def create_camera(
    camera_index=None,
    width=DEFAULT_CAMERA_FRAME_WIDTH,
    height=DEFAULT_CAMERA_FRAME_HEIGHT,
    camera_source=None,
    rotation=None,
    flip=None,
    color_order=None,
):
    camera_index = DEFAULT_CAMERA_INDEX if camera_index is None else camera_index
    camera_source = (camera_source or DEFAULT_CAMERA_SOURCE).strip().lower()
    rotation = DEFAULT_CAMERA_ROTATION if rotation is None else rotation
    flip = (flip or DEFAULT_CAMERA_FLIP).strip().lower()
    color_order = _normalize_color_order(color_order or DEFAULT_CAMERA_COLOR_ORDER)

    if camera_source in {"picamera2", "picamera", "libcamera"}:
        return Picamera2Camera(
            width=width,
            height=height,
            rotation=rotation,
            flip=flip,
            color_order=color_order,
        )

    if camera_source == "auto" and is_raspberry_pi() and Picamera2 is not None:
        try:
            return Picamera2Camera(
                width=width,
                height=height,
                rotation=rotation,
                flip=flip,
                color_order=color_order,
            )
        except Exception:
            pass

    return _open_opencv_camera(
        camera_index=camera_index,
        width=width,
        height=height,
        rotation=rotation,
        flip=flip,
    )
