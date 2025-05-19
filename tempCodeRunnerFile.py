import os
import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models import ModelManager
import config
from tqdm import tqdm

yolo, clf, facenet, device, unique_labels = ModelManager.get_yolo(), joblib.load(config.SVM_MODEL_PATH), *ModelManager.get_facenet(), np.unique(np.load(config.LABELS_PATH)).tolist()

simple_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.FACENET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.FACENET_MEAN, std=config.FACENET_STD)
])

def is_good_quality(frame, img_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = (config.QUALITY_BRIGHTNESS_RANGE[0] < brightness < config.QUALITY_BRIGHTNESS_RANGE[1] and
               sharpness > config.QUALITY_SHARPNESS_THRESHOLD)
    if not is_good:
        print(f"Ảnh {img_path}: Khuôn mặt bị loại - Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")
    return is_good

def prep_face(img, device, img_path):
    if img is None or img.size == 0:
        print(f"Ảnh {img_path}: Ảnh đầu vào trống hoặc lỗi")
        return None
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = simple_transform(img_rgb)
        if not torch.isfinite(img_tensor).all():
            print(f"Ảnh {img_path}: Tensor chứa NaN/Inf")
            return None
        return img_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Ảnh {img_path}: Lỗi tiền xử lý ảnh: {e}")
        return None

def load_embeddings():
    try:
        embeddings = np.load(config.EMBEDDINGS_PATH)
        labels = np.load(config.LABELS_PATH)
        label_to_mean_emb = {}
        for label in np.unique(labels):
            mask = labels == label
            label_to_mean_emb[label] = np.mean(embeddings[mask], axis=0)
    except Exception as e:
        print(f"Lỗi khi tải embeddings/labels: {e}")
        label_to_mean_emb = {}
    return label_to_mean_emb

def test_face_recognition():
    label_to_mean_emb = load_embeddings()
    true_labels = []
    pred_labels = []
    image_names = []
    unknown_count = 0
    uncertain_count = 0
    correct_count = 0
    total_images = 0
    person_count = 0
    known_person_count = 0
    unknown_person_count = 0

    for person_name in os.listdir(config.TEST_IMAGES_DIR):
        person_dir = os.path.join(config.TEST_IMAGES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        person_count += 1
        if person_name in unique_labels:
            known_person_count += 1
        else:
            unknown_person_count += 1
        total_images += len([img for img in os.listdir(person_dir) if img.lower().endswith(('.jpg', '.png'))])

    print(f"Tổng số người trong tập test: {person_count}")
    print(f"Số người có trong dataset: {known_person_count}")
    print(f"Số người không có trong dataset (Unknown): {unknown_person_count}")
    print(f"Tổng số ảnh cần xử lý: {total_images}")

    if not os.path.exists(config.TEST_IMAGES_DIR):
        print(f"Lỗi: Thư mục {config.TEST_IMAGES_DIR} không tồn tại!")
        return

    progress = tqdm(total=total_images, desc="Xử lý ảnh")
    for person_name in os.listdir(config.TEST_IMAGES_DIR):
        person_dir = os.path.join(config.TEST_IMAGES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(person_dir, img_name)
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    raise ValueError("Không đọc được ảnh")
            except Exception as e:
                print(f"Lỗi đọc ảnh {img_path}: {e}")
                progress.update(1)
                continue

            results = yolo(frame)
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            h, w, _ = frame.shape
            if len(coords) > 1:
                print(f"Cảnh báo: Ảnh {img_path} có {len(coords)} khuôn mặt. Chỉ xử lý khuôn mặt đầu tiên.")
            if len(coords) == 0:
                print(f"Ảnh {img_path}: Không tìm thấy khuôn mặt")
                progress.update(1)
                continue

            x1, y1, x2, y2 = coords[0]
            cls = classes[0]
            if cls != 0:
                print(f"Ảnh {img_path}: Đối tượng không phải khuôn mặt (class {cls})")
                progress.update(1)
                continue

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))

            if (x2 - x1) < config.MIN_FACE_SIZE or (y2 - y1) < config.MIN_FACE_SIZE:
                print(f"Ảnh {img_path}: Khuôn mặt quá nhỏ ({x2-x1}x{y2-y1})")
                progress.update(1)
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0 or not is_good_quality(face, img_path):
                print(f"Ảnh {img_path}: Vùng khuôn mặt không hợp lệ hoặc chất lượng thấp")
                progress.update(1)
                continue

            img_tensor = prep_face(face, device, img_path)
            if img_tensor is None:
                print(f"Ảnh {img_path}: Lỗi tiền xử lý khuôn mặt")
                progress.update(1)
                continue

            with torch.no_grad():
                emb = facenet(img_tensor).squeeze().detach().cpu().numpy().reshape(1, -1)
            if np.all(emb == 0) or not np.isfinite(emb).all():
                print(f"Ảnh {img_path}: Embedding không hợp lệ")
                progress.update(1)
                continue

            prob = clf.predict_proba(emb).max()
            predicted_label = clf.predict(emb)[0]

            similarities = {}
            max_similarity = 0.0
            best_label = None
            if label_to_mean_emb:
                for label, mean_emb in label_to_mean_emb.items():
                    sim = cosine_similarity(emb, mean_emb.reshape(1, -1))[0][0]
                    similarities[label] = sim
                    if sim > max_similarity:
                        max_similarity = sim
                        best_label = label

            if max_similarity < config.SVM_COSINE_THRESHOLD_UNKNOWN and prob < config.SVM_PROB_THRESHOLD:
                label = 'Unknown'
                unknown_count += 1
                print(f"Ảnh {img_path}: Gắn nhãn Unknown (Similarity: {max_similarity:.3f}, Prob: {prob:.3f})")
            elif prob < config.SVM_PROB_THRESHOLD and max_similarity < config.SVM_COSINE_THRESHOLD and max_similarity > config.SVM_COSINE_THRESHOLD_UNKNOWN:
                label = 'Uncertain'
                uncertain_count += 1
                print(f"Ảnh {img_path}: Gắn nhãn Uncertain (Similarity: {max_similarity:.3f}, Prob: {prob:.3f})")
            else:
                label = predicted_label
                print(f"Ảnh {img_path}: Gắn nhãn {label} (Similarity: {max_similarity:.3f}, Prob: {prob:.3f})")

            true_label = person_name if person_name in unique_labels else 'Unknown'
            true_labels.append(true_label)
            pred_labels.append(label)
            image_names.append(img_path)

            if label == true_label:
                correct_count += 1
            else:
                print(f"Dự đoán sai: {img_path}, True: {true_label}, Pred: {label}")

            progress.update(1)

    progress.close()

    print("\n=== Kết quả đánh giá ===")
    print(f"Tổng số người trong tập test: {person_count}")
    print(f"Số người có trong dataset: {known_person_count}")
    print(f"Số người không có trong dataset (Unknown): {unknown_person_count}")
    print(f"Tổng số ảnh xử lý: {len(image_names)}")
    if true_labels and pred_labels:
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Độ chính xác (Accuracy): {accuracy:.3f}")
        print(f"Số ảnh dự đoán đúng: {correct_count}")
        print(f"Số ảnh Unknown: {unknown_count}")
        print(f"Số ảnh Uncertain: {uncertain_count}")
        print(f"Số ảnh nhận diện được (nhãn hợp lệ): {len(true_labels) - unknown_count - uncertain_count}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))
        print("\nConfusion Matrix:")
        all_labels = sorted(set(true_labels + pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
        print(f"Labels: {all_labels}")
        print(cm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
        plt.title('Confusion Matrix Heatmap', fontsize=14)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.tight_layout()

        try:
            os.makedirs(config.CHART_DIR, exist_ok=True)
            if not os.access(config.CHART_DIR, os.W_OK):
                raise PermissionError(f"Không có quyền ghi vào thư mục {config.CHART_DIR}/!")
            heatmap_path = os.path.join(config.CHART_DIR, "confusion_matrix_heatmap.png")
            plt.savefig(heatmap_path)
            print(f"Đã lưu biểu đồ heatmap vào '{heatmap_path}'")
        except Exception as e:
            print(f"Lỗi khi lưu biểu đồ heatmap: {e}")
        plt.close()

    else:
        print("Không có kết quả để đánh giá!")

if __name__ == "__main__":
    test_face_recognition()