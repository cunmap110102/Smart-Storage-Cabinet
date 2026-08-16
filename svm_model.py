import os
import pickle
import cv2
import numpy as np

SVM_STATE_MODEL_FILE = "svm_state_model.pickle"

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except Exception:
    LabelEncoder = None
    SVC = None
    SKLEARN_AVAILABLE = False

def extract_features_for_state(face_crop):
    """
    Extract lightweight features from face crop for state classification.
    Using a resized flattened image + gradient magnitude.
    """
    if face_crop.size == 0:
        return None
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    
    # Simple HOG-like feature using Sobel
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Flatten magnitude and resized image
    features = np.concatenate([resized.flatten() / 255.0, magnitude.flatten() / 255.0])
    return features

def train_state_svm(face_crops, labels, output_file=SVM_STATE_MODEL_FILE):
    """
    Train SVM for Face State Classification (normal, mask, glasses, mask_glasses).
    """
    if not SKLEARN_AVAILABLE:
        return None

    x_train = []
    y_train_labels = []
    
    for crop, label in zip(face_crops, labels):
        feats = extract_features_for_state(crop)
        if feats is not None:
            x_train.append(feats)
            y_train_labels.append(label)
            
    if len(x_train) < 2 or len(set(y_train_labels)) < 2:
        return None

    x_train = np.asarray(x_train)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_labels)

    classifier = SVC(kernel="linear", probability=True, class_weight="balanced")
    classifier.fit(x_train, y_train)

    payload = {
        "classifier": classifier,
        "label_encoder": encoder,
        "labels": encoder.classes_.tolist(),
    }
    with open(output_file, "wb") as file_obj:
        pickle.dump(payload, file_obj)
    return payload

def load_state_svm(model_file=SVM_STATE_MODEL_FILE):
    if not SKLEARN_AVAILABLE or not os.path.exists(model_file):
        return None
    try:
        with open(model_file, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception:
        return None

def predict_face_state_svm(face_crop, model_payload):
    state, _probabilities = predict_face_state_svm_details(face_crop, model_payload)
    return state


def predict_face_state_svm_details(face_crop, model_payload):
    if not model_payload or face_crop.size == 0:
        return "normal", {"normal": 1.0}  # fallback to normal
    
    feats = extract_features_for_state(face_crop)
    if feats is None:
        return "normal", {"normal": 1.0}
        
    classifier = model_payload["classifier"]
    label_encoder = model_payload["label_encoder"]
    
    probabilities = classifier.predict_proba([feats])[0]
    best_index = int(np.argmax(probabilities))
    predicted_state = str(label_encoder.inverse_transform([best_index])[0])
    labels = list(label_encoder.classes_)
    probability_map = {str(label): float(prob) for label, prob in zip(labels, probabilities)}
    return predicted_state, probability_map
