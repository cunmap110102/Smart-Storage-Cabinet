import threading
import time


class RelayController:
    def __init__(
        self,
        locker_pins,
        pulse_seconds=3.0,
        active_low=True,
        door_sensor_pins=None,
        door_sensor_closed_active_low=True,
        stop_relay_when_door_opens=True,
        min_pulse_seconds_before_open_check=0.8,
        door_open_settle_seconds=0.05,
        pulse_poll_seconds=0.02,
    ):
        self.locker_pins = dict(locker_pins)
        self.pulse_seconds = float(pulse_seconds)
        self.active_low = active_low
        self.door_sensor_pins = dict(door_sensor_pins or {})
        self.door_sensor_closed_active_low = door_sensor_closed_active_low
        self.stop_relay_when_door_opens = stop_relay_when_door_opens
        self.min_pulse_seconds_before_open_check = float(min_pulse_seconds_before_open_check)
        self.door_open_settle_seconds = float(door_open_settle_seconds)
        self.pulse_poll_seconds = float(pulse_poll_seconds)
        self._gpio = None
        self._is_initialized = False
        self._lock = threading.RLock()

    def _load_gpio(self):
        if self._gpio is not None:
            return self._gpio

        try:
            import RPi.GPIO as gpio
        except ImportError as exc:
            raise RuntimeError(
                "Khong tim thay thu vien RPi.GPIO. Can cai tren Raspberry Pi de dieu khien relay."
            ) from exc

        self._gpio = gpio
        return gpio

    def _active_state(self):
        gpio = self._load_gpio()
        return gpio.LOW if self.active_low else gpio.HIGH

    def _inactive_state(self):
        gpio = self._load_gpio()
        return gpio.HIGH if self.active_low else gpio.LOW

    def setup(self):
        with self._lock:
            if self._is_initialized:
                return

            gpio = self._load_gpio()
            gpio.setwarnings(False)
            gpio.setmode(gpio.BCM)

            inactive_state = self._inactive_state()
            for pin in self.locker_pins.values():
                gpio.setup(pin, gpio.OUT, initial=inactive_state)

            pull_mode = gpio.PUD_UP if self.door_sensor_closed_active_low else gpio.PUD_DOWN
            for pin in self.door_sensor_pins.values():
                gpio.setup(pin, gpio.IN, pull_up_down=pull_mode)

            self._is_initialized = True

    def pulse_locker(self, locker_id, seconds=None):
        with self._lock:
            if locker_id not in self.locker_pins:
                raise ValueError(f"Locker ID khong hop le: {locker_id}")

            self.setup()
            gpio = self._load_gpio()
            pin = self.locker_pins[locker_id]
            active_state = self._active_state()
            inactive_state = self._inactive_state()
            if seconds is None and locker_id in self.door_sensor_pins:
                duration = None
            else:
                duration = self.pulse_seconds if seconds is None else float(seconds)

            gpio.output(pin, active_state)
            try:
                return self._wait_locker_pulse(locker_id, duration)
            finally:
                gpio.output(pin, inactive_state)

    def open_locker(self, locker_id, seconds=None, on_complete=None):
        def worker():
            error = None
            try:
                self.pulse_locker(locker_id, seconds=seconds)
            except Exception as exc:
                error = exc

            if on_complete is not None:
                on_complete(error)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def has_door_sensor(self, locker_id):
        return locker_id in self.door_sensor_pins

    def is_door_closed(self, locker_id):
        with self._lock:
            if locker_id not in self.door_sensor_pins:
                raise ValueError(f"Locker ID chua khai bao cam bien cua: {locker_id}")

            self.setup()
            gpio = self._load_gpio()
            pin = self.door_sensor_pins[locker_id]
            value = gpio.input(pin)
            if self.door_sensor_closed_active_low:
                return value == gpio.LOW
            return value == gpio.HIGH

    def _is_door_closed_stable(self, locker_id, settle_seconds=0.05):
        first_read = self.is_door_closed(locker_id)
        time.sleep(max(0.0, float(settle_seconds)))
        return first_read and self.is_door_closed(locker_id)

    def _is_door_open_stable(self, locker_id, settle_seconds=0.05):
        first_read = not self.is_door_closed(locker_id)
        time.sleep(max(0.0, float(settle_seconds)))
        return first_read and not self.is_door_closed(locker_id)

    def _wait_locker_pulse(self, locker_id, duration):
        if (
            not self.stop_relay_when_door_opens
            or locker_id not in self.door_sensor_pins
        ):
            time.sleep(max(0.0, self.pulse_seconds if duration is None else duration))
            return {"stopped_by_door_open": False}

        deadline = None if duration is None else time.monotonic() + max(0.0, duration)
        poll_delay = max(0.01, self.pulse_poll_seconds)
        started_closed = self._is_door_closed_stable(
            locker_id,
            settle_seconds=self.door_open_settle_seconds,
        )
        if not started_closed:
            return {"stopped_by_door_open": False, "started_closed": False}

        min_hold_seconds = max(
            0.0,
            self.min_pulse_seconds_before_open_check
            if duration is None
            else min(duration, self.min_pulse_seconds_before_open_check),
        )
        min_hold_deadline = time.monotonic() + min_hold_seconds
        while time.monotonic() < min_hold_deadline:
            time.sleep(min(poll_delay, max(0.0, min_hold_deadline - time.monotonic())))

        while deadline is None or time.monotonic() < deadline:
            if self._is_door_open_stable(
                locker_id,
                settle_seconds=self.door_open_settle_seconds,
            ):
                return {"stopped_by_door_open": True, "started_closed": True}
            if duration is None:
                time.sleep(poll_delay)
            else:
                time.sleep(min(poll_delay, max(0.0, deadline - time.monotonic())))

        return {"stopped_by_door_open": False, "started_closed": True}

    def wait_for_door_open(
        self,
        locker_id,
        wait_open_seconds=10.0,
        poll_seconds=0.1,
    ):
        if locker_id not in self.door_sensor_pins:
            return {"sensor": False, "opened_seen": False, "closed": None}

        poll_delay = max(0.02, float(poll_seconds))
        if self._is_door_open_stable(locker_id):
            return {"sensor": True, "opened_seen": True, "closed": False}

        open_deadline = time.monotonic() + max(0.0, float(wait_open_seconds))
        while time.monotonic() < open_deadline:
            time.sleep(poll_delay)
            if self._is_door_open_stable(locker_id):
                return {"sensor": True, "opened_seen": True, "closed": False}

        raise TimeoutError(f"Chua nhan duoc tin hieu mo cua tu so {locker_id}.")

    def wait_for_door_closed(
        self,
        locker_id,
        wait_close_seconds=60.0,
        poll_seconds=0.1,
    ):
        if locker_id not in self.door_sensor_pins:
            return {"sensor": False, "closed": None}

        poll_delay = max(0.02, float(poll_seconds))
        close_deadline = time.monotonic() + max(0.0, float(wait_close_seconds))
        while time.monotonic() < close_deadline:
            if self._is_door_closed_stable(locker_id):
                return {"sensor": True, "closed": True}
            time.sleep(poll_delay)

        raise TimeoutError(f"Chua nhan duoc tin hieu dong cua tu so {locker_id}.")

    def wait_for_door_closed_after_open(
        self,
        locker_id,
        wait_open_seconds=10.0,
        wait_close_seconds=60.0,
        poll_seconds=0.1,
        on_open_detected=None,
    ):
        open_result = self.wait_for_door_open(
            locker_id,
            wait_open_seconds=wait_open_seconds,
            poll_seconds=poll_seconds,
        )
        if not open_result.get("sensor"):
            return {"sensor": False, "opened_seen": False, "closed": None}

        if on_open_detected is not None:
            on_open_detected(open_result)

        close_result = self.wait_for_door_closed(
            locker_id,
            wait_close_seconds=wait_close_seconds,
            poll_seconds=poll_seconds,
        )
        return {
            "sensor": True,
            "opened_seen": bool(open_result.get("opened_seen")),
            "closed": close_result.get("closed"),
        }

    def cleanup(self):
        with self._lock:
            if not self._is_initialized:
                return

            gpio = self._load_gpio()
            inactive_state = self._inactive_state()
            for pin in self.locker_pins.values():
                gpio.output(pin, inactive_state)
            gpio.cleanup()
            self._is_initialized = False
