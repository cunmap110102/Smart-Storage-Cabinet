import argparse
import sys
import time


LOCKER_SENSOR_PIN_MAP = {
    "01": 25,
    "02": 17,
    "03": 22,
}

DOOR_SENSOR_CLOSED_ACTIVE_LOW = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Theo doi lien tuc 3 cam bien cua MC-38 tren Raspberry Pi."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Chu ky doc cam bien (giay). Mac dinh: 0.2",
    )
    parser.add_argument(
        "--snapshot-every",
        type=float,
        default=1.0,
        help="Chu ky in lai toan bo trang thai (giay). Mac dinh: 1.0",
    )
    return parser.parse_args()


def load_gpio():
    try:
        import RPi.GPIO as gpio
    except ImportError as exc:
        raise RuntimeError(
            "Khong tim thay thu vien RPi.GPIO. Hay chay file nay tren Raspberry Pi "
            "va cai bang lenh: pip install RPi.GPIO"
        ) from exc
    return gpio


def setup_gpio(gpio):
    gpio.setwarnings(False)
    gpio.setmode(gpio.BCM)
    pull_mode = gpio.PUD_UP if DOOR_SENSOR_CLOSED_ACTIVE_LOW else gpio.PUD_DOWN
    for pin in LOCKER_SENSOR_PIN_MAP.values():
        gpio.setup(pin, gpio.IN, pull_up_down=pull_mode)


def read_sensor_state(gpio, pin):
    raw_value = gpio.input(pin)
    is_closed = raw_value == gpio.LOW if DOOR_SENSOR_CLOSED_ACTIVE_LOW else raw_value == gpio.HIGH
    raw_text = "LOW" if raw_value == gpio.LOW else "HIGH"
    state_text = "CLOSED" if is_closed else "OPEN"
    return {
        "raw_value": raw_value,
        "raw_text": raw_text,
        "is_closed": is_closed,
        "state_text": state_text,
    }


def format_status_line(states):
    parts = []
    for locker_id in sorted(states):
        state = states[locker_id]
        parts.append(
            f"Tu {locker_id} (GPIO {LOCKER_SENSOR_PIN_MAP[locker_id]}): "
            f"{state['state_text']} [{state['raw_text']}]"
        )
    return " | ".join(parts)


def monitor_loop(gpio, interval, snapshot_every):
    previous_states = {}
    last_snapshot_at = 0.0

    print("Bat dau theo doi 3 cam bien MC-38. Nhan Ctrl+C de dung.")
    print(
        "Quy uoc hien tai: "
        f"{'LOW = CLOSED, HIGH = OPEN' if DOOR_SENSOR_CLOSED_ACTIVE_LOW else 'HIGH = CLOSED, LOW = OPEN'}"
    )

    while True:
        now = time.monotonic()
        states = {
            locker_id: read_sensor_state(gpio, pin)
            for locker_id, pin in LOCKER_SENSOR_PIN_MAP.items()
        }

        changed_lockers = [
            locker_id
            for locker_id, state in states.items()
            if locker_id not in previous_states
            or state["raw_value"] != previous_states[locker_id]["raw_value"]
        ]

        if changed_lockers:
            change_note = ", ".join(f"Tu {locker_id}" for locker_id in changed_lockers)
            print(f"[{time.strftime('%H:%M:%S')}] Thay doi: {change_note}")
            print(format_status_line(states))
            last_snapshot_at = now
        elif now - last_snapshot_at >= snapshot_every:
            print(f"[{time.strftime('%H:%M:%S')}] {format_status_line(states)}")
            last_snapshot_at = now

        previous_states = states
        time.sleep(max(0.05, interval))


def main():
    args = parse_args()
    gpio = load_gpio()
    setup_gpio(gpio)

    try:
        monitor_loop(
            gpio=gpio,
            interval=max(0.05, float(args.interval)),
            snapshot_every=max(0.2, float(args.snapshot_every)),
        )
    except KeyboardInterrupt:
        print("\nDa dung theo doi cam bien.")
    except Exception as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            gpio.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
