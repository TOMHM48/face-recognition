import cv2
import os
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from models import ModelManager
import config

yolo = ModelManager.get_yolo()
INPUT_DIR = config.AUGMENTED_IMAGES_DIR
OUTPUT_DIR = config.CROPPED_IMAGES_DIR

def is_good_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = (config.QUALITY_BRIGHTNESS_RANGE[0] < brightness < config.QUALITY_BRIGHTNESS_RANGE[1] and
               sharpness > config.QUALITY_SHARPNESS_THRESHOLD)
    if not is_good:
        print(f"Khuôn mặt bị loại - Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")
    return is_good

def process_image(img_path, person_folder, out_person_dir, person):
    face_candidates = []
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Không đọc được ảnh")
        results = yolo(img)
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, classes):
            if int(cls) != 0 or conf < config.YOLO_CONF_THRESHOLD:
                continue
            face_width, face_height = x2 - x1, y2 - y1
            if face_width < config.MIN_FACE_SIZE or face_height < config.MIN_FACE_SIZE:
                print(f"Khuôn mặt quá nhỏ: {os.path.basename(img_path)} ({face_width}x{face_height})")
                continue
            face = img[int(y1):int(y2), int(x1):int(x2)]
            if face.size == 0 or face.shape[0] < config.MIN_FACE_SIZE or face.shape[1] < config.MIN_FACE_SIZE:
                print(f"Khuôn mặt sau khi cắt quá nhỏ: {os.path.basename(img_path)}")
                continue
            if not is_good_quality(face):
                continue
            face_candidates.append((face, os.path.basename(img_path)))
    except Exception as e:
        print(f"Lỗi xử lý {img_path}: {e}")
    return face_candidates

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"Lỗi: Thư mục {INPUT_DIR} không tồn tại!")
        exit()
    persons = [p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))]
    if not persons:
        print(f"Lỗi: Không tìm thấy thư mục người nào trong {INPUT_DIR}")
        exit()
    
    for person in persons:
        person_folder = os.path.join(INPUT_DIR, person)
        out_person_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(out_person_dir, exist_ok=True)
        
        face_candidates = []
        image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.png'))]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_image, os.path.join(person_folder, fname), person_folder, out_person_dir, person) 
                       for fname in image_files]
            for future in futures:
                face_candidates.extend(future.result())
        
        img_count = 0
        if len(face_candidates) < config.TARGET_CROPPED_IMAGES:
            print(f"Cảnh báo: Chỉ có {len(face_candidates)} ảnh hợp lệ cho {person}, không đủ {config.TARGET_CROPPED_IMAGES} ảnh.")
        else:
            face_candidates = random.sample(face_candidates, config.TARGET_CROPPED_IMAGES)
        
        for face, fname in face_candidates:
            img_count += 1
            save_path = os.path.join(out_person_dir, f"{person}_{img_count}.jpg")
            try:
                cv2.imwrite(save_path, face)
                print(f"Đã lưu khuôn mặt: {save_path}")
            except Exception as e:
                print(f"Lỗi lưu khuôn mặt {save_path}: {e}")
        
        print(f"Đã cắt {img_count} khuôn mặt cho {person}")
        if img_count < config.TARGET_CROPPED_IMAGES:
            print(f"Cảnh báo: Chỉ cắt được {img_count} ảnh cho {person}. Nên kiểm tra ảnh trong 'datasets/augmented_images' hoặc thu thập thêm ảnh gốc.")
    
    print("Hoàn tất cắt khuôn mặt vào 'datasets/cropped_images/'!")

if __name__ == "__main__":
    main()