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
    FaceRecognitionSession,
    classify_face_state,
    load_state_svm,
)

OUTPUT_DIRNAME = "diagnostics"
CSV_FILENAME = "mode3_recognition_results.csv"
SUMMARY_FILENAME = "mode3_recognition_summary.md"
PASS_CONFIDENCE_THRESHOLD = 68.0
REVIEW_CONFIDENCE_THRESHOLD = 58.0
DISTANCE_THRESHOLD = 0.42


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Mode 3 recognition diagnostics.")
    parser.add_argument("--expected-name", default="", help="Expected person/locker id")
    parser.add_argument("--condition", default="unspecified", help="Condition label")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout per trial")
    parser.add_argument("--min-display", type=float, default=3.0, help="Stable display time per trial")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD, help="Embedding distance threshold")
    parser.add_argument("--output-dir", default=OUTPUT_DIRNAME, help="Output folder")
    return parser.parse_args()


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


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


def categorize_result(final_name, final_confidence):
    if not final_name or final_name == "Unknown":
        return "unknown_or_rejected"
    if final_confidence >= PASS_CONFIDENCE_THRESHOLD:
        return "pass_direct"
    if final_confidence >= REVIEW_CONFIDENCE_THRESHOLD:
        return "review_zone"
    return "low_confidence_rejected"


def draw_overlay(frame, lines):
    overlay = frame.copy()
    box_height = 28 + max(0, len(lines) - 1) * 24
    cv2.rectangle(overlay, (0, 0), (660, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        y += 24
    return frame


def create_session(args):
    session = FaceRecognitionSession(
        threshold=args.threshold,
        timeout_seconds=args.timeout,
        min_display_seconds=args.min_display,
        mode="take",
    )
    session.prediction_distances = []
    session.last_prediction_distance = None
    original_predict_identity = session._predict_identity

    def wrapped_predict_identity(*wrapped_args, **wrapped_kwargs):
        result = original_predict_identity(*wrapped_args, **wrapped_kwargs)
        distance = result.get("distance")
        session.last_prediction_distance = distance
        if distance is not None:
            session.prediction_distances.append(float(distance))
        return result

    session._predict_identity = wrapped_predict_identity
    return session


def run_trial(cap, args, trial_index, state_svm_model):
    session = create_session(args)
    started_at = time.time()
    last_packet = None
    observed_states = Counter()
    dominant_state = "unknown"
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame from camera.")

        packet = session.process_frame(frame)
        last_packet = packet
        if len(session.boxes) == 1:
            face_state = classify_face_state(frame, session.boxes[0], state_svm_model)
            observed_states[face_state] += 1
            dominant_state = observed_states.most_common(1)[0][0]
        display = packet["frame"].copy()
        elapsed = time.time() - started_at
        lines = [
            f"Recognition diagnostics | trial {trial_index}/{args.trials}",
            f"Expected={args.expected_name or '-'} | Condition={args.condition}",
            f"Elapsed: {elapsed:.1f}s | Detector={packet['backend_text']} | State={dominant_state}",
            f"Status: {packet['status_text']}",
            "Press q to stop",
        ]
        draw_overlay(display, lines)
        cv2.imshow("Mode 3 Recognition Diagnostics", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if packet["done"]:
            break

    result = (last_packet or {}).get("result") or {}
    final_name = result.get("name", "Unknown")
    final_confidence = float(result.get("confidence", 0.0))
    expected_name = args.expected_name.strip()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trial_index": trial_index,
        "condition": args.condition,
        "expected_name": expected_name,
        "final_name": final_name,
        "final_confidence": format_float(final_confidence, 1),
        "final_backend": result.get("backend", ""),
        "dominant_state": dominant_state,
        "observed_state_counts": "; ".join(f"{k}={v}" for k, v in sorted(observed_states.items())),
        "recognition_attempts": session.recognition_attempt_count,
        "confirmed_match_frames": session.confirmed_match_frames,
        "distance_mean": format_float(safe_mean(session.prediction_distances), 4),
        "distance_min": format_float(safe_min(session.prediction_distances), 4),
        "distance_max": format_float(safe_max(session.prediction_distances), 4),
        "mode3_decision": categorize_result(final_name, final_confidence),
        "expected_match": (
            "yes" if expected_name and final_name == expected_name else "no" if expected_name else ""
        ),
    }


def write_csv_rows(csv_path, rows):
    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    decision_counts = Counter(row["mode3_decision"] for row in rows)
    lines = [
        "# Mode 3 Recognition Summary",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Trials: {len(rows)}",
        f"- Pass direct: {decision_counts.get('pass_direct', 0)}",
        f"- Review zone: {decision_counts.get('review_zone', 0)}",
        f"- Unknown or rejected: {decision_counts.get('unknown_or_rejected', 0)}",
        "",
        "| Trial | Condition | Expected | Final | Conf | Distance Mean | Decision | Match |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['trial_index']} | {row['condition']} | {row['expected_name']} | "
            f"{row['final_name']} | {row['final_confidence']} | {row['distance_mean']} | "
            f"{row['mode3_decision']} | {row['expected_match']} |"
        )
    return "\n".join(lines) + "\n"


def print_trial_table(row):
    columns = [
        "trial_index",
        "condition",
        "expected_name",
        "dominant_state",
        "final_name",
        "final_confidence",
        "distance_mean",
        "recognition_attempts",
        "confirmed_match_frames",
        "mode3_decision",
        "expected_match",
    ]
    print("", flush=True)
    print("Recognition trial", flush=True)
    print(render_terminal_table([row], columns), flush=True)


def print_summary_tables(rows):
    columns = [
        "trial_index",
        "condition",
        "expected_name",
        "dominant_state",
        "final_name",
        "final_confidence",
        "distance_mean",
        "mode3_decision",
        "expected_match",
    ]
    print("", flush=True)
    print("Recognition run summary", flush=True)
    print(render_terminal_table(rows, columns), flush=True)

    decision_counts = Counter(row["mode3_decision"] for row in rows)
    match_counts = Counter(row["expected_match"] for row in rows if row["expected_match"])
    state_counts = Counter(row["dominant_state"] for row in rows if row["dominant_state"])
    conf_values = [float(row["final_confidence"]) for row in rows if row["final_confidence"]]
    dist_values = [float(row["distance_mean"]) for row in rows if row["distance_mean"]]
    aggregate_row = {
        "trials": str(len(rows)),
        "pass_direct": str(decision_counts.get("pass_direct", 0)),
        "review_zone": str(decision_counts.get("review_zone", 0)),
        "unknown_or_rejected": str(decision_counts.get("unknown_or_rejected", 0)),
        "expected_yes": str(match_counts.get("yes", 0)),
        "expected_no": str(match_counts.get("no", 0)),
        "conf_mean": format_float(safe_mean(conf_values), 2) or "-",
        "dist_mean": format_float(safe_mean(dist_values), 4) or "-",
        "states": "; ".join(f"{k}={state_counts[k]}" for k in sorted(state_counts)) or "-",
    }
    print("", flush=True)
    print("Recognition aggregate", flush=True)
    print(
        render_terminal_table(
            [aggregate_row],
            [
                "trials",
                "pass_direct",
                "review_zone",
                "unknown_or_rejected",
                "expected_yes",
                "expected_no",
                "conf_mean",
                "dist_mean",
                "states",
            ],
        ),
        flush=True,
    )


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
    state_svm_model = load_state_svm()

    rows = []
    cv2.namedWindow("Mode 3 Recognition Diagnostics")
    try:
        for trial_index in range(1, args.trials + 1):
            row = run_trial(cap, args, trial_index, state_svm_model)
            rows.append(row)
            print_trial_table(row)
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("Recognition diagnostics stopped by user.", file=sys.stderr, flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not rows:
        print("No completed recognition trials were recorded.", flush=True)
        return

    write_csv_rows(csv_path, rows)
    summary_text = build_summary(rows)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    print_summary_tables(rows)
    print("", flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
