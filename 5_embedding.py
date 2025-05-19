import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from models import ModelManager
import config

DATASET_DIR = config.CROPPED_IMAGES_DIR
EMB_FILE = config.EMBEDDINGS_PATH
LABEL_FILE = config.LABELS_PATH

facenet, device = ModelManager.get_facenet()

if not os.path.exists(DATASET_DIR):
    print(f"Lỗi: Thư mục {DATASET_DIR} không tồn tại!")
    exit()
persons = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]
if len(persons) < 2:
    print(f"Lỗi: Cần ít nhất 2 nhãn, hiện tại chỉ có {len(persons)}!")
    exit()
for person in persons:
    person_dir = os.path.join(DATASET_DIR, person)
    images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    print(f"{person}: {len(images)} ảnh")
    if len(images) < config.MIN_IMAGES_PER_PERSON:
        print(f"Cảnh báo: Số lượng ảnh cho {person} quá ít ({len(images)}).")

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.FACENET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.FACENET_MEAN, std=config.FACENET_STD)
])

def preprocess_image(img):
    if img is None or img.size == 0:
        print("Ảnh đầu vào trống hoặc lỗi")
        return None
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = data_transforms(img_rgb)
        if not torch.isfinite(img_tensor).all():
            print("Tensor chứa giá trị không hợp lệ (NaN hoặc Inf)")
            return None
        return img_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh: {e}")
        return None

def extract_embeddings(dataset_dir, batch_size=config.BATCH_SIZE):
    embeddings = []
    labels = []
    batch_images = []
    batch_labels = []
    skipped_images = 0
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            print(f"Bỏ qua: {person_dir} không phải thư mục")
            continue
        print(f"Xử lý nhãn: {person_name}")
        for img_name in os.listdir(person_dir):
            if not img_name.endswith('.jpg'):
                continue
            img_path = os.path.join(person_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Không đọc được ảnh")
            except Exception as e:
                print(f"Lỗi đọc ảnh: {img_path} - {e}")
                skipped_images += 1
                continue
            if img.shape[0] < config.MIN_IMAGE_SIZE[0] or img.shape[1] < config.MIN_IMAGE_SIZE[1]:
                print(f"Ảnh quá nhỏ: {img_path}")
                skipped_images += 1
                continue
            img_tensor = preprocess_image(img)
            if img_tensor is None:
                print(f"Lỗi tiền xử lý ảnh: {img_path}")
                skipped_images += 1
                continue
            batch_images.append(img_tensor)
            batch_labels.append(person_name)
            if len(batch_images) >= batch_size:
                batch_tensor = torch.cat(batch_images, dim=0)
                with torch.no_grad():
                    batch_emb = facenet(batch_tensor).cpu().numpy()
                for emb, lbl in zip(batch_emb, batch_labels):
                    if np.all(emb == 0) or not np.isfinite(emb).all():
                        print(f"Embedding không hợp lệ: {img_path}")
                        skipped_images += 1
                        continue
                    if emb.shape[0] != 512:
                        print(f"Embedding sai kích thước: {img_path}, shape: {emb.shape}")
                        skipped_images += 1
                        continue
                    embeddings.append(emb)
                    labels.append(lbl)
                batch_images, batch_labels = [], []
    
    if batch_images:
        batch_tensor = torch.cat(batch_images, dim=0)
        with torch.no_grad():
            batch_emb = facenet(batch_tensor).cpu().numpy()
        for emb, lbl in zip(batch_emb, batch_labels):
            if np.all(emb == 0) or not np.isfinite(emb).all():
                print(f"Embedding không hợp lệ: {img_path}")
                skipped_images += 1
                continue
            if emb.shape[0] != 512:
                print(f"Embedding sai kích thước: {img_path}, shape: {emb.shape}")
                skipped_images += 1
                continue
            embeddings.append(emb)
            labels.append(lbl)
    
    if not embeddings:
        raise ValueError("Không có embedding nào được tạo! Kiểm tra dữ liệu hoặc mô hình.")
    
    print(f"Đã bỏ qua {skipped_images} ảnh do lỗi hoặc không hợp lệ")
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    label_counts = Counter(labels)
    print("Số lượng embeddings cho mỗi nhãn:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
        if count < config.MIN_IMAGES_PER_PERSON:
            print(f"Cảnh báo: Số lượng embeddings cho {label} quá ít ({count}).")
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Lỗi: Chỉ có 1 nhãn, không đủ để huấn luyện phân loại!")
        exit()
    
    print("\nPhân tích độ tương đồng (cosine similarity):")
    for i, label1 in enumerate(unique_labels):
        mask1 = labels == label1
        emb1 = embeddings[mask1]
        mean_emb1 = np.mean(emb1, axis=0)
        for label2 in unique_labels[i+1:]:
            mask2 = labels == label2
            emb2 = embeddings[mask2]
            mean_emb2 = np.mean(emb2, axis=0)
            similarity = cosine_similarity([mean_emb1], [mean_emb2])[0][0]
            print(f"{label1} vs {label2}: {similarity:.3f}")
    
    print("\nĐộ phân tán embeddings trong mỗi nhãn:")
    for label in unique_labels:
        mask = labels == label
        emb = embeddings[mask]
        if len(emb) > 1:
            intra_sim = cosine_similarity(emb).mean()
            print(f"{label}: Trung bình cosine similarity trong nhãn: {intra_sim:.3f}")
    
    return embeddings, labels

if __name__ == "__main__":
    print("Đang trích xuất embeddings...")
    try:
        embeddings, labels = extract_embeddings(DATASET_DIR)
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        if not os.access(config.EMBEDDINGS_DIR, os.W_OK):
            print(f"Lỗi: Không có quyền ghi vào thư mục {config.EMBEDDINGS_DIR}/!")
            exit()
        np.save(EMB_FILE, embeddings)
        np.save(LABEL_FILE, labels)
        print(f"Đã lưu {len(labels)} embeddings vào '{EMB_FILE}' và labels vào '{LABEL_FILE}'.")
    except Exception as e:
        print(f"Lỗi khi trích xuất embeddings: {e}")