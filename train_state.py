import os
import cv2

from capture_data import detect_face_boxes
from train_model import (
    create_synthetic_masked_variant,
    create_synthetic_glasses_variant,
    create_synthetic_mask_glasses_variant,
    is_synthetic_image_file,
)
from svm_model import train_state_svm

def train_state_classifier(dataset_dir="dataset"):
    face_crops = []
    labels = []

    if not os.path.exists(dataset_dir):
        print(f"Khong tim thay thu muc {dataset_dir}")
        return

    print("Dang trich xuat du lieu de huan luyen SVM phan loai trang thai (mat tran / kinh / khau trang)...")
    for user_name in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_dir):
            continue
            
        for filename in os.listdir(user_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            file_name_lower = filename.lower()
            if (
                is_synthetic_image_file(filename)
                or "_mask_" in file_name_lower
                or "_glasses_" in file_name_lower
            ):
                continue
                
            image_path = os.path.join(user_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, _ = detect_face_boxes(rgb_image, prefer_yunet=False)
            
            for box in boxes:
                top, right, bottom, left = box
                
                # 1. Trang thai Normal
                crop_normal = image[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
                if crop_normal.size > 0:
                    face_crops.append(crop_normal)
                    labels.append("normal")
                    
                # 2. Trang thai Mask (Tao nhieu phien ban mau khau trang de SVM nhan dien thuc te)
                for color_type in ["gray", "black", "white", "blue"]:
                    img_mask = create_synthetic_masked_variant(image, box, color_type)
                    if img_mask is not None:
                        crop_mask = img_mask[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
                        if crop_mask.size > 0:
                            face_crops.append(crop_mask)
                            labels.append("mask")
                        
                # 3. Trang thai Glasses
                img_glasses = create_synthetic_glasses_variant(image, box)
                if img_glasses is not None:
                    crop_glasses = img_glasses[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
                    if crop_glasses.size > 0:
                        face_crops.append(crop_glasses)
                        labels.append("glasses")

                # 4. Trang thai Mask + Glasses
                for color_type in ["gray", "black"]:
                    img_mg = create_synthetic_mask_glasses_variant(image, box, color_type)
                    if img_mg is not None:
                        crop_mg = img_mg[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
                        if crop_mg.size > 0:
                            face_crops.append(crop_mg)
                            labels.append("mask_glasses")

    if len(face_crops) > 0:
        print(f"Dang huan luyen voi {len(face_crops)} mau crops...")
        train_state_svm(face_crops, labels)
        print("Huan luyen xong! Da tao file 'svm_state_model.pickle' thanh cong.")
    else:
        print("Khong tim thay du lieu khuon mat de huan luyen. Hay dang ky it nhat 1 nguoi dung truoc!")

if __name__ == "__main__":
    train_state_classifier()
