import json
import os
import pickle
import shutil
import tempfile
import threading

import cv2
import face_recognition
import numpy as np

from capture_data import detect_face_boxes
from eye_band_features import get_full_upper_encoding

MODEL_WRITE_LOCK = threading.RLock()
SYNTHETIC_MASK_COLOR = (236, 236, 236)
SYNTHETIC_GLASSES_COLOR = (32, 32, 32)
SYNTHETIC_FILE_MARKER = "_synthetic_"
TRAIN_MAX_IMAGE_WIDTH = 640


def is_synthetic_image_file(filename):
    return SYNTHETIC_FILE_MARKER in filename.lower()


def save_synthetic_image(user_dir, source_filename, variant_name, image, box_index=0):
    source_base, _source_ext = os.path.splitext(source_filename)
    output_filename = f"{source_base}_synthetic_{variant_name}_{box_index + 1}.jpg"
    output_path = os.path.join(user_dir, output_filename)

    if os.path.exists(output_path):
        return output_path

    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"Cannot save synthetic image: {output_path}")

    return output_path


def resize_for_training(image, max_width=TRAIN_MAX_IMAGE_WIDTH):
    height, width = image.shape[:2]
    if width <= max_width:
        return image, 1.0

    scale = max_width / float(width)
    resized = cv2.resize(
        image,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def scale_face_box(face_box, scale):
    if scale == 1.0:
        return tuple(int(value) for value in face_box)
    return tuple(int(round(value * scale)) for value in face_box)


def load_cached_face_boxes(user_dir, filename, scale=1.0):
    metadata_path = os.path.join(user_dir, os.path.splitext(filename)[0] + ".json")
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        face_box = metadata.get("face_box")
        if not isinstance(face_box, list) or len(face_box) != 4:
            return None
        return [scale_face_box(face_box, scale)]
    except Exception:
        return None


def unscale_face_box(face_box, scale):
    if scale == 1.0:
        return tuple(int(value) for value in face_box)
    return tuple(int(round(value / scale)) for value in face_box)


def save_cached_face_boxes(user_dir, filename, image, boxes, scale=1.0):
    if not boxes:
        return

    metadata_path = os.path.join(user_dir, os.path.splitext(filename)[0] + ".json")
    if os.path.exists(metadata_path):
        return

    try:
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "image_file": filename,
                    "image_shape": list(image.shape[:2]),
                    "face_box": list(unscale_face_box(boxes[0], scale)),
                    "detector": "train_fallback",
                    "capture_mode": "unknown",
                },
                metadata_file,
            )
    except Exception:
        pass

def enhance_lighting(rgb_image):
    """Can bang sang thich ung (CLAHE) giup on dinh model goc."""
    try:
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception:
        return rgb_image

def create_synthetic_masked_variant(image, face_box, color_type="gray"):
    top, right, bottom, left = face_box
    face_width = right - left
    face_height = bottom - top
    if face_width < 30 or face_height < 30:
        return None

    masked_image = image.copy()
    upper_y = top + int(face_height * 0.44)
    lower_y = top + int(face_height * 0.90)
    cheek_inset = max(4, int(face_width * 0.08))
    chin_inset = max(8, int(face_width * 0.16))
    mask_points = np.array(
        [
            [left + cheek_inset, upper_y],
            [right - cheek_inset, upper_y],
            [right - chin_inset, lower_y],
            [left + chin_inset, lower_y],
        ],
        dtype=np.int32,
    )
    
    if color_type == "black":
        color = (30, 30, 30)
        edge_color = (10, 10, 10)
    elif color_type == "white":
        color = (245, 245, 245)
        edge_color = (200, 200, 200)
    elif color_type == "blue":
        color = (215, 175, 145)
        edge_color = (190, 150, 120)
    else:
        color = SYNTHETIC_MASK_COLOR
        edge_color = (210, 210, 210)

    cv2.fillConvexPoly(masked_image, mask_points, color)
    cv2.polylines(masked_image, [mask_points], True, edge_color, 1, cv2.LINE_AA)
    return masked_image


def create_synthetic_glasses_variant(image, face_box):
    top, right, bottom, left = face_box
    face_width = right - left
    face_height = bottom - top
    if face_width < 30 or face_height < 30:
        return None

    glasses_image = image.copy()
    lens_top = top + int(face_height * 0.28)
    lens_bottom = top + int(face_height * 0.48)
    left_lens_left = left + int(face_width * 0.10)
    left_lens_right = left + int(face_width * 0.43)
    right_lens_left = right - int(face_width * 0.43)
    right_lens_right = right - int(face_width * 0.10)
    bridge_left = left + int(face_width * 0.45)
    bridge_right = right - int(face_width * 0.45)
    temple_y = top + int(face_height * 0.36)

    lens_overlay = glasses_image.copy()
    cv2.rectangle(
        lens_overlay,
        (left_lens_left, lens_top),
        (left_lens_right, lens_bottom),
        (74, 74, 74),
        -1,
    )
    cv2.rectangle(
        lens_overlay,
        (right_lens_left, lens_top),
        (right_lens_right, lens_bottom),
        (74, 74, 74),
        -1,
    )
    cv2.addWeighted(lens_overlay, 0.28, glasses_image, 0.72, 0, glasses_image)

    cv2.rectangle(
        glasses_image,
        (left_lens_left, lens_top),
        (left_lens_right, lens_bottom),
        SYNTHETIC_GLASSES_COLOR,
        2,
    )
    cv2.rectangle(
        glasses_image,
        (right_lens_left, lens_top),
        (right_lens_right, lens_bottom),
        SYNTHETIC_GLASSES_COLOR,
        2,
    )
    cv2.line(
        glasses_image,
        (bridge_left, (lens_top + lens_bottom) // 2),
        (bridge_right, (lens_top + lens_bottom) // 2),
        SYNTHETIC_GLASSES_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        glasses_image,
        (left_lens_left, temple_y),
        (max(0, left - int(face_width * 0.10)), temple_y - 2),
        SYNTHETIC_GLASSES_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        glasses_image,
        (right_lens_right, temple_y),
        (min(image.shape[1] - 1, right + int(face_width * 0.10)), temple_y - 2),
        SYNTHETIC_GLASSES_COLOR,
        2,
        cv2.LINE_AA,
    )
    return glasses_image


def create_synthetic_mask_glasses_variant(image, face_box, color_type="gray"):
    masked_image = create_synthetic_masked_variant(image, face_box, color_type)
    if masked_image is None:
        return None
    return create_synthetic_glasses_variant(masked_image, face_box)


def append_encodings_for_image(encodings, names, rgb_image, boxes, user_name, include_eye_band=True):
    enhanced_rgb = enhance_lighting(rgb_image)
    image_encodings = face_recognition.face_encodings(enhanced_rgb, boxes)
    added_count = 0
    upper_feature_count = 0
    upper_feature_errors = 0
    for i, encoding in enumerate(image_encodings):
        encodings.append(encoding)
        names.append(user_name)
        added_count += 1
        
        # Chi them upper-face feature cho cac mau can ho tro nhan dien khi bi che phan duoi mat.
        if include_eye_band and len(boxes) > i:
            try:
                # Chuyen RGB sang BGR de trich xuat upper face
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                upper_face_encoding = get_full_upper_encoding(boxes[i], bgr_image)
                if upper_face_encoding.size > 0:
                    # Ket hop face encoding voi upper face encoding
                    combined_encoding = np.concatenate([encoding, upper_face_encoding])
                    encodings.append(combined_encoding)
                    names.append(user_name)
                    added_count += 1
                    upper_feature_count += 1
            except Exception:
                upper_feature_errors += 1
    return added_count, upper_feature_count, upper_feature_errors


def load_model(output_file="encodings.pickle"):
    if not os.path.exists(output_file):
        return {"encodings": [], "names": []}

    with open(output_file, "rb") as file_obj:
        return pickle.load(file_obj)


def save_model(data, output_file="encodings.pickle"):
    output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    temp_file = None
    temp_path = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=output_dir,
            prefix="encodings_",
            suffix=".tmp",
        )
        temp_path = temp_file.name
        pickle.dump(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        temp_file = None
        os.replace(temp_path, output_file)
        temp_path = None
    finally:
        if temp_file is not None:
            temp_file.close()
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_user_encodings(user_name, dataset_dir="dataset"):
    user_dir = os.path.join(dataset_dir, user_name)
    if not os.path.isdir(user_dir):
        raise FileNotFoundError(f"User folder '{user_dir}' does not exist.")

    encodings = []
    names = []
    image_files = [
        filename
        for filename in os.listdir(user_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")) and not is_synthetic_image_file(filename)
    ]
    success_count = 0
    upper_feature_count = 0
    upper_feature_errors = 0
    source_stats = {
        "original": {"images": 0, "encodings": 0},
        "glasses": {"images": 0, "encodings": 0},
        "mask": {"images": 0, "encodings": 0},
        "mask_glasses": {"images": 0, "encodings": 0},
    }

    print(f"\nProcessing user: {user_name}", flush=True)
    print(f"Found {len(image_files)} files", flush=True)

    cached_box_hits = 0
    detector_fallbacks = 0

    for filename in image_files:
        image_path = os.path.join(user_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  [WARN] Cannot read image: {image_path}", flush=True)
            continue

        train_image, image_scale = resize_for_training(image)
        rgb_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
        boxes = load_cached_face_boxes(user_dir, filename, scale=image_scale)
        if boxes is not None:
            cached_box_hits += 1
        else:
            detector_fallbacks += 1
            boxes, _backend = detect_face_boxes(rgb_image, prefer_yunet=False)
            save_cached_face_boxes(user_dir, filename, image, boxes, scale=image_scale)

        if not boxes:
            print(f"  [SKIP] No face found in: {filename}", flush=True)
            continue

        added_count, upper_count, upper_errors = append_encodings_for_image(
            encodings,
            names,
            rgb_image,
            boxes,
            user_name,
            include_eye_band=False,
        )
        success_count += added_count
        upper_feature_count += upper_count
        upper_feature_errors += upper_errors
        source_stats["original"]["images"] += 1
        source_stats["original"]["encodings"] += added_count

        file_name_lower = filename.lower()
        should_create_synthetic_files = True

        if should_create_synthetic_files and "_mask_" not in file_name_lower:
            for box_index, box in enumerate(boxes):
                synthetic_mask_image = create_synthetic_masked_variant(train_image, box)
                if synthetic_mask_image is None:
                    continue
                save_synthetic_image(user_dir, filename, "mask", synthetic_mask_image, box_index)

                synthetic_rgb_image = cv2.cvtColor(synthetic_mask_image, cv2.COLOR_BGR2RGB)
                added_count, upper_count, upper_errors = append_encodings_for_image(
                    encodings,
                    names,
                    synthetic_rgb_image,
                    [box],
                    user_name,
                    include_eye_band=True,
                )
                success_count += added_count
                upper_feature_count += upper_count
                upper_feature_errors += upper_errors
                source_stats["mask"]["images"] += 1
                source_stats["mask"]["encodings"] += added_count

        if should_create_synthetic_files and "_glasses_" not in file_name_lower:
            for box_index, box in enumerate(boxes):
                synthetic_glasses_image = create_synthetic_glasses_variant(train_image, box)
                if synthetic_glasses_image is None:
                    continue
                save_synthetic_image(user_dir, filename, "glasses", synthetic_glasses_image, box_index)

                synthetic_glasses_rgb = cv2.cvtColor(synthetic_glasses_image, cv2.COLOR_BGR2RGB)
                added_count, upper_count, upper_errors = append_encodings_for_image(
                    encodings,
                    names,
                    synthetic_glasses_rgb,
                    [box],
                    user_name,
                    include_eye_band=False,
                )
                success_count += added_count
                upper_feature_count += upper_count
                upper_feature_errors += upper_errors
                source_stats["glasses"]["images"] += 1
                source_stats["glasses"]["encodings"] += added_count

        if (
            should_create_synthetic_files
            and "_mask_glasses_" not in file_name_lower
            and "_mask_" not in file_name_lower
        ):
            for box_index, box in enumerate(boxes):
                synthetic_mask_glasses_image = create_synthetic_mask_glasses_variant(train_image, box)
                if synthetic_mask_glasses_image is None:
                    continue
                save_synthetic_image(
                    user_dir,
                    filename,
                    "mask_glasses",
                    synthetic_mask_glasses_image,
                    box_index,
                )

                synthetic_mask_glasses_rgb = cv2.cvtColor(synthetic_mask_glasses_image, cv2.COLOR_BGR2RGB)
                added_count, upper_count, upper_errors = append_encodings_for_image(
                    encodings,
                    names,
                    synthetic_mask_glasses_rgb,
                    [box],
                    user_name,
                    include_eye_band=True,
                )
                success_count += added_count
                upper_feature_count += upper_count
                upper_feature_errors += upper_errors
                source_stats["mask_glasses"]["images"] += 1
                source_stats["mask_glasses"]["encodings"] += added_count

    print("Train source summary:", flush=True)
    print(
        f"  - Face box cache: {cached_box_hits} hit, {detector_fallbacks} detect fallback",
        flush=True,
    )
    print(
        "  - Anh goc mat thuong: "
        f"{source_stats['original']['images']} anh, "
        f"{source_stats['original']['encodings']} encodings",
        flush=True,
    )
    print(
        "  - Anh gia lap deo kinh: "
        f"{source_stats['glasses']['images']} anh, "
        f"{source_stats['glasses']['encodings']} encodings",
        flush=True,
    )
    print(
        "  - Anh gia lap deo khau trang: "
        f"{source_stats['mask']['images']} anh, "
        f"{source_stats['mask']['encodings']} encodings",
        flush=True,
    )
    print(
        "  - Anh gia lap deo kinh + khau trang: "
        f"{source_stats['mask_glasses']['images']} anh, "
        f"{source_stats['mask_glasses']['encodings']} encodings",
        flush=True,
    )

    print(
        f"Completed {user_name}: extracted {success_count}/{len(image_files)} usable samples. "
        f"Upper-feature samples: {upper_feature_count}, errors: {upper_feature_errors}.",
        flush=True,
    )
    return encodings, names


def extract_original_face_vectors(user_name, dataset_dir="dataset"):
    user_dir = os.path.join(dataset_dir, user_name)
    if not os.path.isdir(user_dir):
        raise FileNotFoundError(f"User folder '{user_dir}' does not exist.")

    face_vectors = []
    for filename in os.listdir(user_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if "_mask_" in filename.lower() or "_glasses_" in filename.lower() or is_synthetic_image_file(filename):
            continue

        image_path = os.path.join(user_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        train_image, image_scale = resize_for_training(image)
        rgb_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
        boxes = load_cached_face_boxes(user_dir, filename, scale=image_scale)
        if boxes is None:
            boxes, _backend = detect_face_boxes(rgb_image, prefer_yunet=False)
            save_cached_face_boxes(user_dir, filename, image, boxes, scale=image_scale)
        if not boxes:
            continue

        enhanced_rgb = enhance_lighting(rgb_image)
        encodings = face_recognition.face_encodings(enhanced_rgb, boxes)
        for encoding in encodings:
            vector = np.asarray(encoding, dtype=np.float64)[:128]
            if vector.shape[0] == 128:
                face_vectors.append(vector)

    return face_vectors


def inspect_model_consistency(dataset_dir="dataset", output_file="encodings.pickle"):
    dataset_users = set()
    if os.path.isdir(dataset_dir):
        for folder_name in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder_name)
            if os.path.isdir(folder_path):
                dataset_users.add(folder_name)

    data = load_model(output_file)
    model_users = set(data.get("names", []))
    encoding_lengths = sorted({len(encoding) for encoding in data.get("encodings", [])})

    return {
        "dataset_users": dataset_users,
        "model_users": model_users,
        "missing_in_model": sorted(dataset_users - model_users),
        "missing_in_dataset": sorted(model_users - dataset_users),
        "encoding_lengths": encoding_lengths,
        "has_upper_features": any(length > 128 for length in encoding_lengths),
        "encoding_count": len(data.get("encodings", [])),
    }


def inspect_identity_separation(output_file="encodings.pickle", top_k=5):
    data = load_model(output_file)
    raw_encodings = data.get("encodings", [])
    raw_names = data.get("names", [])

    grouped_vectors = {}
    for encoding, name in zip(raw_encodings, raw_names):
        vector = np.asarray(encoding, dtype=np.float64)
        if vector.shape[0] < 128:
            continue
        if vector.shape[0] != 128:
            continue
        grouped_vectors.setdefault(str(name), []).append(vector[:128])

    users = sorted(grouped_vectors)
    pair_rows = []
    nearest_neighbors = {}

    for user_name in users:
        best_neighbor = None
        best_distance = None
        source_vectors = grouped_vectors[user_name]
        if not source_vectors:
            continue
        source_matrix = np.vstack(source_vectors)

        for other_name in users:
            if other_name == user_name:
                continue
            target_vectors = grouped_vectors[other_name]
            if not target_vectors:
                continue
            target_matrix = np.vstack(target_vectors)
            distances = np.linalg.norm(
                source_matrix[:, None, :] - target_matrix[None, :, :],
                axis=2,
            )
            min_distance = float(np.min(distances))
            mean_distance = float(np.mean(distances))

            if best_distance is None or min_distance < best_distance:
                best_distance = min_distance
                best_neighbor = other_name

            if user_name < other_name:
                pair_rows.append(
                    {
                        "user_a": user_name,
                        "user_b": other_name,
                        "min_distance": min_distance,
                        "mean_distance": mean_distance,
                        "sample_count_a": int(source_matrix.shape[0]),
                        "sample_count_b": int(target_matrix.shape[0]),
                    }
                )

        if best_neighbor is not None and best_distance is not None:
            nearest_neighbors[user_name] = {
                "neighbor": best_neighbor,
                "min_distance": float(best_distance),
            }

    pair_rows.sort(key=lambda row: row["min_distance"])
    risk_pairs = []
    for row in pair_rows[: max(0, int(top_k))]:
        risk_level = "ok"
        if row["min_distance"] < 0.38:
            risk_level = "high"
        elif row["min_distance"] < 0.45:
            risk_level = "medium"
        risk_pairs.append(
            {
                **row,
                "risk_level": risk_level,
            }
        )

    return {
        "user_count": len(users),
        "base_encoding_user_count": len(grouped_vectors),
        "nearest_neighbors": nearest_neighbors,
        "risk_pairs": risk_pairs,
    }


def remove_user_from_model(user_name, output_file="encodings.pickle"):
    with MODEL_WRITE_LOCK:
        data = load_model(output_file)
        filtered_encodings = []
        filtered_names = []
        removed_count = 0

        for encoding, name in zip(data.get("encodings", []), data.get("names", [])):
            if name == user_name:
                removed_count += 1
                continue
            filtered_encodings.append(encoding)
            filtered_names.append(name)

        updated_data = {"encodings": filtered_encodings, "names": filtered_names}
        save_model(updated_data, output_file)
        return removed_count, updated_data


def train_user(user_name, dataset_dir="dataset", output_file="encodings.pickle", replace_existing=True):
    print(f"--- Incremental training for '{user_name}' ---")

    with MODEL_WRITE_LOCK:
        new_encodings, new_names = extract_user_encodings(user_name, dataset_dir)
        if not new_encodings:
            raise RuntimeError(f"No usable face samples found for '{user_name}'.")

        data = load_model(output_file)
        current_encodings = data.get("encodings", [])
        current_names = data.get("names", [])

        if replace_existing:
            current_encodings = [
                encoding for encoding, name in zip(current_encodings, current_names) if name != user_name
            ]
            current_names = [name for name in current_names if name != user_name]

        current_encodings.extend(new_encodings)
        current_names.extend(new_names)

        updated_data = {"encodings": current_encodings, "names": current_names}
        save_model(updated_data, output_file)

    print(
        f"COMPLETE! Added {len(new_encodings)} encodings for '{user_name}'. "
        f"Model now has {len(current_encodings)} total encodings."
    )
    return updated_data


def delete_user(user_name, dataset_dir="dataset", output_file="encodings.pickle", delete_dataset=True):
    with MODEL_WRITE_LOCK:
        removed_count, updated_data = remove_user_from_model(user_name, output_file)

        if delete_dataset:
            user_dir = os.path.join(dataset_dir, user_name)
            if os.path.isdir(user_dir):
                shutil.rmtree(user_dir)

    print(f"Deleted user '{user_name}' with {removed_count} encodings removed from model.")
    return removed_count, updated_data


def train_model(dataset_dir="dataset", output_file="encodings.pickle"):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset folder '{dataset_dir}' does not exist.")

    print("--- Full training ---")
    with MODEL_WRITE_LOCK:
        data = {"encodings": [], "names": []}
        save_model(data, output_file)

        user_folders = [
            folder_name
            for folder_name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, folder_name))
        ]
        print(f"Found {len(user_folders)} user folders: {user_folders}")

        for name in user_folders:
            train_user(name, dataset_dir=dataset_dir, output_file=output_file, replace_existing=False)

        final_data = load_model(output_file)
    print(f"\nCOMPLETE! Saved {len(final_data['encodings'])} encodings to '{output_file}'.")
    return final_data


def main():
    train_model()


if __name__ == "__main__":
    main()
