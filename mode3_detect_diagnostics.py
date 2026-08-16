import argparse
import csv
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime

import cv2

from camera_utils import create_camera
from recognize_face import (
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    FRAME_PROCESS_SCALE,
    classify_face_state,
    detect_face_boxes,
    get_face_width_ratio,
    is_face_inside_guide,
    load_state_svm,
    measure_face_quality,
)

OUTPUT_DIRNAME = "diagnostics"
CSV_FILENAME = "mode3_detect_results.csv"
SUMMARY_FILENAME = "mode3_detect_summary.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Mode 3 detector diagnostics.")
    parser.add_argument("--condition", default="unspecified", help="Condition label, e.g. low_light_front")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to observe")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--output-dir", default=OUTPUT_DIRNAME, help="Output folder")
    return parser.parse_args()


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def get_guide_rect(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    guide_width = int(frame_width * 0.40)
    guide_height = int(frame_height * 0.55)
    left = (frame_width - guide_width) // 2
    top = (frame_height - guide_height) // 2
    return left, top, left + guide_width, top + guide_height


def format_float(value, digits=2):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def safe_mean(values):
    if not values:
        return None
    return statistics.fmean(values)


def safe_min(values):
    if not values:
        return None
    return min(values)


def safe_max(values):
    if not values:
        return None
    return max(values)


def render_terminal_table(rows, columns):
    if not rows:
        return ""
    normalized_rows = [{column: str(row.get(column, "")) for column in columns} for row in rows]
    widths = {column: max(len(column), *(len(row[column]) for row in normalized_rows)) for column in columns}

    def sep():
        return "+-" + "-+-".join("-" * widths[column] for column in columns) + "-+"

    def line(row):
        return "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"

    header = {column: column for column in columns}
    out = [sep(), line(header), sep()]
    out.extend(line(row) for row in normalized_rows)
    out.append(sep())
    return "\n".join(out)


def normalize_backend_name(backend_text):
    base = str(backend_text or "").split(":", 1)[0].strip()
    return base or "Unknown"


def flatten_counter(counter_obj):
    if not counter_obj:
        return ""
    return "; ".join(f"{key}={counter_obj[key]}" for key in sorted(counter_obj))


def draw_overlay(frame, lines):
    overlay = frame.copy()
    box_height = 28 + max(0, len(lines) - 1) * 24
    cv2.rectangle(overlay, (0, 0), (640, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        y += 24
    return frame


def write_csv_row(csv_path, row):
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_summary(row):
    lines = [
        "# Mode 3 Detect Summary",
        "",
        f"- Generated at: {row['timestamp']}",
        f"- Condition: {row['condition']}",
        f"- Duration seconds: {row['duration_seconds']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in row.items():
        if key in {"timestamp", "condition", "duration_seconds"}:
            continue
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, CSV_FILENAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILENAME)

    cap = create_camera(
        camera_index=args.camera_index,
        width=CAMERA_FRAME_WIDTH,
        height=CAMERA_FRAME_HEIGHT,
    )
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    detector_counts = Counter()
    state_counts = Counter()
    brightness_values = []
    sharpness_values = []
    face_width_ratios = []
    no_face_frames = 0
    multi_face_frames = 0
    single_face_frames = 0
    aligned_frames = 0
    misaligned_frames = 0
    low_light_frames = 0
    blur_frames = 0
    too_far_frames = 0
    too_close_frames = 0
    accessory_frames = 0
    accepted_normal_frames = 0
    frame_count = 0
    state_svm_model = load_state_svm()

    started_at = time.time()
    cv2.namedWindow("Mode 3 Detect Diagnostics")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read frame from camera.")

            frame_count += 1
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_PROCESS_SCALE, fy=FRAME_PROCESS_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            raw_boxes, backend_text = detect_face_boxes(rgb_small_frame)
            scaled_boxes = [
                (
                    int(round(top / FRAME_PROCESS_SCALE)),
                    int(round(right / FRAME_PROCESS_SCALE)),
                    int(round(bottom / FRAME_PROCESS_SCALE)),
                    int(round(left / FRAME_PROCESS_SCALE)),
                )
                for top, right, bottom, left in raw_boxes
            ]
            detector_counts[normalize_backend_name(backend_text)] += 1

            guide_rect = get_guide_rect(frame.shape)
            if len(scaled_boxes) == 0:
                no_face_frames += 1
            elif len(scaled_boxes) > 1:
                multi_face_frames += 1
            else:
                single_face_frames += 1
                face_box = scaled_boxes[0]
                face_state = classify_face_state(frame, face_box, state_svm_model)
                state_counts[face_state] += 1
                brightness, sharpness = measure_face_quality(frame, face_box)
                ratio = get_face_width_ratio(frame.shape, face_box)
                brightness_values.append(float(brightness))
                sharpness_values.append(float(sharpness))
                face_width_ratios.append(float(ratio))
                if face_state == "normal":
                    accepted_normal_frames += 1
                else:
                    accessory_frames += 1

                if is_face_inside_guide(face_box, guide_rect):
                    aligned_frames += 1
                else:
                    misaligned_frames += 1
                if brightness < 50.0:
                    low_light_frames += 1
                if sharpness < 25.0:
                    blur_frames += 1
                if ratio < 0.16:
                    too_far_frames += 1
                elif ratio > 0.50:
                    too_close_frames += 1

                top, right, bottom, left = face_box
                box_color = (0, 255, 0) if face_state == "normal" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.putText(
                    frame,
                    f"state={face_state}",
                    (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

            left, top, right, bottom = guide_rect
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 200, 0), 2)
            elapsed = time.time() - started_at
            lines = [
                f"Detect diagnostics | condition={args.condition}",
                f"Elapsed: {elapsed:.1f}/{args.duration:.1f}s | detector={backend_text}",
                f"NoFace={no_face_frames} Multi={multi_face_frames} Single={single_face_frames}",
                f"NormalAccepted={accepted_normal_frames} AccessoryFrames={accessory_frames}",
                "Press q to stop",
            ]
            draw_overlay(frame, lines)
            cv2.imshow("Mode 3 Detect Diagnostics", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or elapsed >= args.duration:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": args.condition,
        "duration_seconds": format_float(time.time() - started_at, 1),
        "frame_count": frame_count,
        "no_face_frames": no_face_frames,
        "multi_face_frames": multi_face_frames,
        "single_face_frames": single_face_frames,
        "aligned_frames": aligned_frames,
        "misaligned_frames": misaligned_frames,
        "low_light_frames": low_light_frames,
        "blur_frames": blur_frames,
        "too_far_frames": too_far_frames,
        "too_close_frames": too_close_frames,
        "accepted_normal_frames": accepted_normal_frames,
        "accessory_frames": accessory_frames,
        "brightness_mean": format_float(safe_mean(brightness_values)),
        "brightness_min": format_float(safe_min(brightness_values)),
        "brightness_max": format_float(safe_max(brightness_values)),
        "sharpness_mean": format_float(safe_mean(sharpness_values)),
        "sharpness_min": format_float(safe_min(sharpness_values)),
        "sharpness_max": format_float(safe_max(sharpness_values)),
        "face_width_ratio_mean": format_float(safe_mean(face_width_ratios), 3),
        "face_width_ratio_min": format_float(safe_min(face_width_ratios), 3),
        "face_width_ratio_max": format_float(safe_max(face_width_ratios), 3),
        "detector_counts": flatten_counter(detector_counts),
        "state_counts": flatten_counter(state_counts),
    }

    write_csv_row(csv_path, row)
    summary_text = build_summary(row)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    columns = [
        "condition",
        "frame_count",
        "no_face_frames",
        "multi_face_frames",
        "single_face_frames",
        "accepted_normal_frames",
        "accessory_frames",
        "low_light_frames",
        "blur_frames",
        "brightness_mean",
        "sharpness_mean",
        "face_width_ratio_mean",
        "detector_counts",
        "state_counts",
    ]
    print("", flush=True)
    print("Detect summary", flush=True)
    print(render_terminal_table([row], columns), flush=True)
    print("", flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
