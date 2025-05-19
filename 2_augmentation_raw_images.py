import gc
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
from concurrent.futures import ProcessPoolExecutor
from models import ModelManager
import config

INPUT_DIR = config.RAW_IMAGES_DIR
OUTPUT_DIR = config.AUGMENTED_IMAGES_DIR

def create_folder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as e:
        print(f"Lỗi khi tạo thư mục {folder_name}: {e}")
        raise

yolo = ModelManager.get_yolo()

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

def ensure_rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("Đã chuyển ảnh grayscale sang RGB")
    return img

def cv2_to_pil(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pil_to_cv2(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise ValueError("Đầu vào phải là ảnh PIL")

def is_good_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = (config.QUALITY_BRIGHTNESS_RANGE[0] < brightness < config.QUALITY_BRIGHTNESS_RANGE[1] and
               sharpness > config.QUALITY_SHARPNESS_THRESHOLD)
    if not is_good:
        print(f"Ảnh bị loại - Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")
    return is_good

def has_face(img):
    try:
        results = yolo(img)
        if results is None or len(results) == 0:
            print("Ảnh bị loại: YOLO không trả về kết quả")
            return False
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, classes):
            if int(cls) == 0 and conf >= config.YOLO_CONF_THRESHOLD and (x2 - x1) >= config.MIN_FACE_SIZE and (y2 - y1) >= config.MIN_FACE_SIZE:
                return True
        print("Ảnh bị loại: Không phát hiện khuôn mặt hoặc khuôn mặt quá nhỏ")
        return False
    except Exception as e:
        print(f"Lỗi khi chạy YOLO: {e}")
        return False

def resize_if_needed(img):
    h, w = img.shape[:2]
    max_h, max_w = (1280, 720)  # Đồng bộ với FRAME_SIZE
    if h > max_h or w > max_w:
        scale = min(max_h / h, max_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

def process_image(img_path, out_person_dir, person_name, idx):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Không đọc được ảnh")
        img = ensure_rgb(img)
        img = resize_if_needed(img)
        if img.shape[0] < config.MIN_IMAGE_SIZE[0] or img.shape[1] < config.MIN_IMAGE_SIZE[1]:
            print(f"Ảnh gốc quá nhỏ: {img_path} ({img.shape[0]}x{img.shape[1]})")
            return None
        if is_good_quality(img) and has_face(img):
            save_path = os.path.join(out_person_dir, f"{person_name}_{idx}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Đã lưu ảnh gốc: {save_path}")
            return img
        print(f"Ảnh gốc không đạt yêu cầu: {img_path}")
        return None
    except Exception as e:
        print(f"Lỗi xử lý {img_path}: {e}")
        return None

def augment_image(img, num_augs=3):
    pil_img = cv2_to_pil(img)
    aug_imgs = []
    for _ in range(num_augs):
        aug_img = augmentation_transforms(pil_img)
        aug_imgs.append(pil_to_cv2(aug_img))
    return aug_imgs

def augment_person(person_folder, out_person_dir, target_images):
    create_folder(out_person_dir)
    image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.png'))]
    original_count = len(image_files)
    
    if original_count == 0:
        print(f"Lỗi: Không tìm thấy ảnh trong {person_folder}")
        return 0
    
    print(f"Xử lý {person_folder}: {original_count} ảnh gốc")
    
    total_images = 0
    valid_original_images = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, os.path.join(person_folder, fname), out_person_dir, os.path.basename(person_folder), idx+1) 
                   for idx, fname in enumerate(image_files)]
        valid_original_images = [f.result() for f in futures if f.result() is not None]
    
    total_images = len(valid_original_images)
    print(f"Số ảnh gốc hợp lệ: {total_images}/{original_count}")
    if not valid_original_images:
        print(f"Lỗi: Không có ảnh gốc hợp lệ trong {person_folder}")
        return total_images
    
    attempts = 0
    max_attempts = 10000
    rejected_reasons = {'quality': 0, 'face': 0, 'size': 0}
    
    while total_images < target_images and attempts < max_attempts:
        img = random.choice(valid_original_images)
        aug_imgs = augment_image(img, num_augs=3)
        
        for aug_img_cv2 in aug_imgs:
            if total_images >= target_images:
                break
            if aug_img_cv2.shape[0] < config.MIN_IMAGE_SIZE[0] or aug_img_cv2.shape[1] < config.MIN_IMAGE_SIZE[1]:
                print(f"Ảnh augmentation không đạt yêu cầu: Kích thước quá nhỏ ({aug_img_cv2.shape[0]}x{aug_img_cv2.shape[1]})")
                rejected_reasons['size'] += 1
                attempts += 1
                continue
            if not is_good_quality(aug_img_cv2):
                rejected_reasons['quality'] += 1
                attempts += 1
                continue
            if not has_face(aug_img_cv2):
                rejected_reasons['face'] += 1
                attempts += 1
                continue
            total_images += 1
            save_path = os.path.join(out_person_dir, f"{os.path.basename(person_folder)}_{total_images}.jpg")
            try:
                cv2.imwrite(save_path, aug_img_cv2)
                print(f"Đã lưu ảnh augmentation: {save_path}")
            except Exception as e:
                print(f"Lỗi lưu ảnh {save_path}: {e}")
            attempts += 1
            gc.collect()
    
    print(f"Đã tạo {total_images} ảnh cho {os.path.basename(out_person_dir)}")
    if total_images < config.TARGET_IMAGES_PER_PERSON:
        print(f"Cảnh báo: Chỉ tạo được {total_images} ảnh. Lý do có thể:")
        print(f"- Ảnh gốc không đủ chất lượng hoặc số lượng ({original_count} ảnh, hợp lệ: {len(valid_original_images)})")
        print(f"- Số ảnh bị loại: Quality={rejected_reasons['quality']}, Face={rejected_reasons['face']}, Size={rejected_reasons['size']}")
        print(f"- Tỷ lệ loại: Quality={rejected_reasons['quality']/attempts*100:.1f}%, Face={rejected_reasons['face']/attempts*100:.1f}%, Size={rejected_reasons['size']/attempts*100:.1f}%")
        print("Gợi ý:")
        print("1. Thu thập thêm ảnh gốc (ít nhất 100 ảnh/người, đa dạng góc độ/ánh sáng)")
        print("2. Sử dụng webcam độ phân giải cao (ít nhất 1280x720)")
    
    return total_images

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Lỗi: Thư mục {INPUT_DIR} không tồn tại!")
        return
    persons = [p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))]
    if not persons:
        print(f"Lỗi: Không tìm thấy thư mục người nào trong {INPUT_DIR}")
        return
    
    for person in persons:
        person_folder = os.path.join(INPUT_DIR, person)
        out_person_dir = os.path.join(OUTPUT_DIR, person)
        augment_person(person_folder, out_person_dir, config.TARGET_IMAGES_PER_PERSON)
    
    print(f"Hoàn tất augmentation! Ảnh được lưu trong {OUTPUT_DIR}")

if __name__ == "__main__":
    main()