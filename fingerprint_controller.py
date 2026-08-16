import os
import time


DEFAULT_FINGERPRINT_BACKEND = os.getenv("FINGERPRINT_BACKEND", "auto").strip().lower()
DEFAULT_FINGERPRINT_PORT = os.getenv("FINGERPRINT_PORT", "/dev/serial0").strip()
DEFAULT_FINGERPRINT_BAUDRATE = int(os.getenv("FINGERPRINT_BAUDRATE", "57600"))
DEFAULT_FINGERPRINT_TIMEOUT_SECONDS = float(os.getenv("FINGERPRINT_TIMEOUT_SECONDS", "10"))


class FingerprintController:
    def __init__(
        self,
        backend=None,
        port=None,
        baudrate=None,
        timeout_seconds=None,
    ):
        self.backend = (backend or DEFAULT_FINGERPRINT_BACKEND).strip().lower()
        self.port = port or DEFAULT_FINGERPRINT_PORT
        self.baudrate = int(baudrate or DEFAULT_FINGERPRINT_BAUDRATE)
        self.timeout_seconds = float(timeout_seconds or DEFAULT_FINGERPRINT_TIMEOUT_SECONDS)

        self.backend_name = "Unavailable"
        self._error_message = ""
        self._serial = None
        self._sensor = None
        self._driver = None

        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "mock":
            self.backend_name = "Mock"
            self._driver = "mock"
            return

        if self.backend not in {"auto", "pyfingerprint", "adafruit"}:
            self._error_message = f"Backend van tay khong duoc ho tro: {self.backend}"
            return

        if self.backend in {"auto", "pyfingerprint"}:
            if self._try_initialize_pyfingerprint():
                return
            if self.backend == "pyfingerprint":
                raise RuntimeError(self._error_message)

        if self.backend in {"auto", "adafruit"}:
            if self._try_initialize_adafruit():
                return
            if self.backend == "adafruit":
                raise RuntimeError(self._error_message)

    def _try_initialize_pyfingerprint(self):
        try:
            from pyfingerprint.pyfingerprint import PyFingerprint
        except ImportError:
            self._error_message = (
                "Thieu thu vien pyfingerprint. Cai bang: pip install pyfingerprint"
            )
            return False

        try:
            sensor = PyFingerprint(
                self.port,
                self.baudrate,
                0xFFFFFFFF,
                0x00000000,
            )
            if not sensor.verifyPassword():
                raise RuntimeError("Sai password cam bien van tay.")
            self._sensor = sensor
            self._driver = "pyfingerprint"
            self.backend_name = "PyFingerprint UART"
            self._error_message = ""
            return True
        except Exception as exc:
            self._sensor = None
            self._driver = None
            self._error_message = f"Khong khoi tao duoc cam bien van tay: {exc}"
            return False

    def _try_initialize_adafruit(self):
        try:
            import serial
        except ImportError:
            self._error_message = "Thieu thu vien pyserial. Cai bang: pip install pyserial"
            return False

        try:
            import adafruit_fingerprint
        except ImportError:
            self._error_message = (
                "Thieu thu vien adafruit-circuitpython-fingerprint. "
                "Cai bang: pip install adafruit-circuitpython-fingerprint"
            )
            return False

        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self._sensor = adafruit_fingerprint.Adafruit_Fingerprint(self._serial)
            self._driver = "adafruit"
            self.backend_name = "Adafruit UART"
            self._error_message = ""
            return True
        except Exception as exc:
            self._sensor = None
            self._driver = None
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._error_message = f"Khong khoi tao duoc cam bien van tay: {exc}"
            return False

    def is_available(self):
        return self.backend_name == "Mock" or self._sensor is not None

    def reinitialize(self):
        if self.is_available():
            return True

        self.cleanup()
        self.backend_name = "Unavailable"
        self._error_message = ""
        self._sensor = None
        self._driver = None
        self._initialize_backend()
        return self.is_available()

    def get_status_text(self):
        if self.is_available():
            return f"Van tay: san sang ({self.backend_name})"
        if self._error_message:
            return f"Van tay: khong san sang ({self._error_message})"
        return "Van tay: khong san sang"

    def cleanup(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    @staticmethod
    def _emit_progress(progress_callback, message):
        if progress_callback is not None:
            progress_callback(message)

    def enroll_locker(self, locker_id, timeout_seconds=None, progress_callback=None):
        template_id = self._locker_to_template_id(locker_id)
        timeout_seconds = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)

        if self.backend_name == "Mock":
            return {
                "matched": True,
                "locker_id": str(locker_id),
                "finger_id": template_id,
                "confidence": 100.0,
                "backend": self.backend_name,
            }

        sensor = self._require_sensor()
        if self._driver == "pyfingerprint":
            self._emit_progress(progress_callback, "Dat ngon tay len cam bien de quet lan 1...")
            self._capture_template_pyfingerprint(sensor, 0x01, timeout_seconds)
            existing_position, _ = sensor.searchTemplate()
            if existing_position >= 0:
                raise RuntimeError(f"Van tay da ton tai o ID {existing_position}.")

            self._emit_progress(progress_callback, "Vui long nhac ngon tay ra, dat lai lan hai...")
            self._wait_for_finger_release_pyfingerprint(sensor, timeout_seconds)
            self._emit_progress(progress_callback, "Dang quet lan 2. Giu ngon tay yen tren cam bien...")
            self._capture_template_pyfingerprint(sensor, 0x02, timeout_seconds)

            if sensor.compareCharacteristics() == 0:
                raise RuntimeError("Hai lan quet van tay khong giong nhau.")

            self._emit_progress(progress_callback, "Dang tao va luu mau van tay...")
            sensor.createTemplate()
            try:
                sensor.deleteTemplate(template_id)
            except Exception:
                pass
            sensor.storeTemplate(template_id)
        else:
            self._emit_progress(progress_callback, "Dat ngon tay len cam bien de quet lan 1...")
            self._capture_template_adafruit(sensor, slot=1, timeout_seconds=timeout_seconds)
            self._emit_progress(progress_callback, "Vui long nhac ngon tay ra, dat lai lan hai...")
            self._wait_for_finger_release_adafruit(sensor, timeout_seconds=timeout_seconds)
            self._emit_progress(progress_callback, "Dang quet lan 2. Giu ngon tay yen tren cam bien...")
            self._capture_template_adafruit(sensor, slot=2, timeout_seconds=timeout_seconds)

            self._emit_progress(progress_callback, "Dang tao va luu mau van tay...")
            result = sensor.create_model()
            if result != self._adafruit_module().OK:
                raise RuntimeError("Khong tao duoc mau van tay. Hay thu lai.")

            delete_model = getattr(sensor, "delete_model", None)
            if delete_model is not None:
                try:
                    delete_model(template_id)
                except Exception:
                    pass

            result = sensor.store_model(template_id)
            if result != self._adafruit_module().OK:
                raise RuntimeError(f"Khong luu duoc mau van tay vao o {template_id}.")

        return {
            "matched": True,
            "locker_id": str(locker_id),
            "finger_id": template_id,
            "confidence": 100.0,
            "backend": self.backend_name,
        }

    def verify_locker(self, expected_locker_id=None, timeout_seconds=None):
        timeout_seconds = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)

        if self.backend_name == "Mock":
            mock_match = os.getenv("FINGERPRINT_MOCK_MATCH", "").strip()
            locker_id = expected_locker_id or mock_match
            if not locker_id:
                return {
                    "matched": False,
                    "locker_id": None,
                    "finger_id": None,
                    "confidence": 0.0,
                    "backend": self.backend_name,
                }

            return {
                "matched": True,
                "locker_id": str(locker_id),
                "finger_id": self._locker_to_template_id(locker_id),
                "confidence": 100.0,
                "backend": self.backend_name,
            }

        sensor = self._require_sensor()
        if self._driver == "pyfingerprint":
            self._capture_template_pyfingerprint(sensor, 0x01, timeout_seconds)
            finger_id, confidence = sensor.searchTemplate()
            if finger_id < 0:
                return {
                    "matched": False,
                    "locker_id": None,
                    "finger_id": None,
                    "confidence": 0.0,
                    "backend": self.backend_name,
                }
        else:
            self._capture_template_adafruit(sensor, slot=1, timeout_seconds=timeout_seconds)
            search_fn = getattr(sensor, "finger_fast_search", None)
            if search_fn is None:
                search_fn = getattr(sensor, "finger_search", None)
            if search_fn is None:
                raise RuntimeError("Thu vien van tay khong ho tro tim kiem mau.")

            result = search_fn()
            if result != self._adafruit_module().OK:
                return {
                    "matched": False,
                    "locker_id": None,
                    "finger_id": None,
                    "confidence": 0.0,
                    "backend": self.backend_name,
                }

            finger_id = int(getattr(sensor, "finger_id", -1))
            confidence = float(getattr(sensor, "confidence", 0.0))

        locker_id = f"{finger_id:02d}"

        if expected_locker_id is not None and str(expected_locker_id) != locker_id:
            return {
                "matched": False,
                "locker_id": locker_id,
                "finger_id": finger_id,
                "confidence": confidence,
                "backend": self.backend_name,
                "error": f"Van tay thuoc tu {locker_id}, khong trung voi khuon mat.",
            }

        return {
            "matched": True,
            "locker_id": locker_id,
            "finger_id": finger_id,
            "confidence": confidence,
            "backend": self.backend_name,
        }

    def delete_locker(self, locker_id):
        template_id = self._locker_to_template_id(locker_id)

        if self.backend_name == "Mock":
            return True

        sensor = self._require_sensor()
        if self._driver == "pyfingerprint":
            return bool(sensor.deleteTemplate(template_id))

        delete_fn = getattr(sensor, "delete_model", None)
        if delete_fn is None:
            raise RuntimeError("Thu vien van tay khong ho tro xoa mau.")

        result = delete_fn(template_id)
        return result == self._adafruit_module().OK

    def clear_all_templates(self):
        if self.backend_name == "Mock":
            return True

        sensor = self._require_sensor()
        if self._driver == "pyfingerprint":
            clear_fn = getattr(sensor, "clearDatabase", None)
            if clear_fn is None:
                clear_fn = getattr(sensor, "emptyDatabase", None)
            if clear_fn is None:
                raise RuntimeError("Thu vien van tay khong ho tro xoa toan bo mau.")
            return bool(clear_fn())

        clear_fn = getattr(sensor, "empty_library", None)
        if clear_fn is None:
            clear_fn = getattr(sensor, "emptyLibrary", None)
        if clear_fn is None:
            raise RuntimeError("Thu vien van tay khong ho tro xoa toan bo mau.")

        result = clear_fn()
        return result == self._adafruit_module().OK

    def _require_sensor(self):
        if self._sensor is None:
            if self._error_message:
                raise RuntimeError(self._error_message)
            raise RuntimeError("Cam bien van tay chua san sang.")
        return self._sensor

    def _capture_template_pyfingerprint(self, sensor, slot, timeout_seconds):
        deadline = time.time() + max(0.1, timeout_seconds)

        while time.time() < deadline:
            if sensor.readImage():
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Het thoi gian cho dat ngon tay len cam bien.")

        sensor.convertImage(slot)

    def _wait_for_finger_release_pyfingerprint(self, sensor, timeout_seconds):
        deadline = time.time() + max(0.1, timeout_seconds)

        while time.time() < deadline:
            if not sensor.readImage():
                return
            time.sleep(0.2)

        raise RuntimeError("Hay nhac ngon tay ra roi dat lai de dang ky lan 2.")

    def _capture_template_adafruit(self, sensor, slot, timeout_seconds):
        deadline = time.time() + max(0.1, timeout_seconds)

        while time.time() < deadline:
            result = sensor.get_image()
            if result == self._adafruit_module().OK:
                break
            if result == self._adafruit_module().NOFINGER:
                time.sleep(0.2)
                continue
            raise RuntimeError("Khong doc duoc anh van tay. Hay lau sach cam bien va thu lai.")
        else:
            raise RuntimeError("Het thoi gian cho dat ngon tay len cam bien.")

        result = sensor.image_2_tz(slot)
        if result != self._adafruit_module().OK:
            raise RuntimeError("Khong chuyen duoc anh van tay thanh mau so.")

    def _wait_for_finger_release_adafruit(self, sensor, timeout_seconds):
        deadline = time.time() + max(0.1, timeout_seconds)

        while time.time() < deadline:
            result = sensor.get_image()
            if result == self._adafruit_module().NOFINGER:
                return
            time.sleep(0.2)

        raise RuntimeError("Hay nhac ngon tay ra roi dat lai de dang ky lan 2.")

    @staticmethod
    def _adafruit_module():
        import adafruit_fingerprint

        return adafruit_fingerprint

    @staticmethod
    def _locker_to_template_id(locker_id):
        try:
            template_id = int(str(locker_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Locker ID khong hop le cho van tay: {locker_id}") from exc

        if template_id < 1:
            raise ValueError(f"Locker ID khong hop le cho van tay: {locker_id}")
        return template_id
