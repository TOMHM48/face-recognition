import cv2
import os
import torch
import numpy as np
import joblib
import random
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import config
 
# Tải các mô hình YOLO, SVM và FaceNet.
def load_models():
    try:
        yolo = YOLO(config.YOLO_MODEL_PATH)
        yolo.conf = config.YOLO_CONF_THRESHOLD
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLO: {e}")
        exit()
 
    try:
        clf = joblib.load(config.SVM_MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi tải mô hình SVM: {e}")
        exit()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    try:
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        if os.path.exists(config.FACENET_FINETUNED_PATH):
            facenet.load_state_dict(torch.load(config.FACENET_FINETUNED_PATH, map_location=device))
            print("Đã tải mô hình FaceNet đã fine-tuned")
        else:
            print("Cảnh báo: Không tìm thấy mô hình fine-tuned. Sử dụng mô hình pretrained.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình FaceNet: {e}")
        exit()
 
    try:
        labels = np.load(config.LABELS_PATH)
        unique_labels = np.unique(labels).tolist()
        if len(unique_labels) < 2:
            print("Lỗi: Cần ít nhất 2 nhãn trong 'labels.npy'!")
            exit()
        print(f"Đã tải {len(unique_labels)} nhãn: {unique_labels}")
    except Exception as e:
        print(f"Lỗi khi tải labels.npy: {e}")
        exit()
 
    if not hasattr(clf, 'classes_'):
        print("Lỗi: Mô hình SVM không có thuộc tính classes_!")
        exit()
    svm_labels = clf.classes_.tolist()
    if sorted(svm_labels) != sorted(unique_labels):
        print(f"Cảnh báo: Nhãn trong SVM ({svm_labels}) không khớp với nhãn trong labels.npy ({unique_labels})!")
        print("Đề xuất: Chạy lại train_svm.py để đảm bảo mô hình SVM khớp với dữ liệu.")
 
    return yolo, clf, facenet, device, unique_labels
 
simple_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.FACENET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.FACENET_MEAN, std=config.FACENET_STD)
])
 
# Kiểm tra chất lượng khuôn mặt.
def is_good_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = (config.QUALITY_BRIGHTNESS_RANGE[0] < brightness < config.QUALITY_BRIGHTNESS_RANGE[1] and
               sharpness > config.QUALITY_SHARPNESS_THRESHOLD)
    if not is_good:
        print(f"Khuôn mặt bị loại - Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")
    return is_good
 
# Tiền xử lý ảnh khuôn mặt.
def prep_face(img, device):
    if img is None or img.size == 0:
        print("Ảnh đầu vào trống hoặc lỗi")
        return None
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = simple_transform(img_rgb)
        if not torch.isfinite(img_tensor).all():
            print("Tensor chứa NaN/Inf")
            return None
        return img_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh: {e}")
        return None
 
label2color = {}
 
# Tạo màu ngẫu nhiên cho nhãn.
def get_color(label):
    if label not in label2color:
        color = tuple(random.randint(0, 255) for _ in range(3))
        label2color[label] = color
    return label2color[label]
 
# Mở webcam khả dụng.
def open_webcam():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
        cap.release()
    print("Lỗi: Không tìm thấy webcam.")
    return None, -1
 
# Tải embeddings để tính cosine similarity.
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
 
# Nhận diện khuôn mặt thời gian thực.
def face_recognition_loop(yolo, clf, facenet, device, unique_labels):
    cap, index = open_webcam()
    if cap is None:
        exit()
 
    print(f"Sử dụng webcam index: {index}")
    frame_counter = 0
    label_to_mean_emb = load_embeddings()
    skip_frames = 1
    last_results = []
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam.")
            break
       
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, config.FRAME_SIZE)
       
        if frame_counter % skip_frames == 0:
            results = yolo(frame)
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
 
            h, w, _ = frame.shape
            print(f"Frame {frame_counter}: Found {len(coords)} faces")
            last_results = []
 
            for (x1, y1, x2, y2), cls in zip(coords, classes):
                if cls != 0:
                    continue
 
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
 
                if (x2 - x1) < config.MIN_FACE_SIZE or (y2 - y1) < config.MIN_FACE_SIZE:
                    print(f"Frame {frame_counter}: Face too small ({x2-x1}x{y2-y1})")
                    continue
 
                face = frame[y1:y2, x1:x2]
                if face.size == 0 or not is_good_quality(face):
                    print(f"Frame {frame_counter}: Invalid face region or low quality")
                    continue
 
                img_tensor = prep_face(face, device)
                if img_tensor is None:
                    print(f"Frame {frame_counter}: Failed to preprocess face")
                    continue
 
                with torch.no_grad():
                    emb = facenet(img_tensor).squeeze().detach().cpu().numpy().reshape(1, -1)
                if np.all(emb == 0) or not np.isfinite(emb).all():
                    print(f"Frame {frame_counter}: Invalid embedding")
                    continue
 
                # Tính xác suất và nhãn dự đoán
                prob = clf.predict_proba(emb).max()
                predicted_label = clf.predict(emb)[0]
                probs = clf.predict_proba(emb)[0]
 
                # Tính độ tương đồng cosine với tất cả các nhãn
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
 
                # In xác suất và độ tương đồng cho từng nhãn
                # print(f"Frame {frame_counter}: Probabilities and Similarities:")
                # for idx, label in enumerate(clf.classes_):
                #     print(f"  {label}: Prob={probs[idx]:.3f}, Sim={similarities.get(label, 0.0):.3f}")
 
                # Logic phân loại
                if max_similarity < config.SVM_COSINE_THRESHOLD_UNKNOWN and prob < config.SVM_PROB_THRESHOLD:
                    label = 'Unknown'
                    prob_display = max_similarity
                    print(f"Frame {frame_counter}: Labeled as Unknown (Max Similarity: {max_similarity:.3f} < {config.SVM_COSINE_THRESHOLD_UNKNOWN}, Max Prob: {prob:.3f} < {config.SVM_PROB_THRESHOLD})")
                elif prob < config.SVM_PROB_THRESHOLD and max_similarity < config.SVM_COSINE_THRESHOLD and max_similarity > config.SVM_COSINE_THRESHOLD_UNKNOWN:
                    label = 'Uncertain'
                    prob_display = prob
                    print(f"Frame {frame_counter}: Labeled as Uncertain (Max Prob: {prob:.3f} < {config.SVM_PROB_THRESHOLD}, Similarity: {max_similarity:.3f} < {config.SVM_COSINE_THRESHOLD})")
                else:
                    label = predicted_label
                    prob_display = prob
                    print(f"Frame {frame_counter}: Labeled as {label} (Max Prob: {prob:.3f} >= {config.SVM_PROB_THRESHOLD}, Max Similarity: {max_similarity:.3f} >= {config.SVM_COSINE_THRESHOLD})")
 
                last_results.append((x1, y1, x2, y2, label, prob_display))
 
        for x1, y1, x2, y2, label, prob in last_results:
            color = get_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {prob:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
 
        cv2.imshow("Face Recognition", frame)
 
        if cv2.waitKey(1) & 0xFF == 27:
            break
       
        frame_counter += 1
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    yolo, clf, facenet, device, unique_labels = load_models()
    face_recognition_loop(yolo, clf, facenet, device, unique_labels)