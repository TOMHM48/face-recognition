import cv2
import os
import re
import numpy as np
from models import ModelManager
import config

# Tạo thư mục nếu chưa tồn tại
def create_folder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as e:
        print(f"Lỗi khi tạo thư mục {folder_name}: {e}")
        raise

# Kiểm tra định dạng tên (chỉ chữ cái, số, dấu cách)
def is_valid_name(name):
    return bool(re.match(r'^[a-zA-Z0-9 ]+$', name))

# Kiểm tra chất lượng ảnh dựa trên độ sáng và sắc nét
def is_good_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = (config.QUALITY_BRIGHTNESS_RANGE[0] < brightness < config.QUALITY_BRIGHTNESS_RANGE[1] and
               sharpness > config.QUALITY_SHARPNESS_THRESHOLD)
    if not is_good:
        print(f"Ảnh chất lượng thấp - Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")
    return is_good

# Tìm và mở webcam khả dụng
def open_webcam():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
        cap.release()
    print("Lỗi: Không tìm thấy webcam. Kiểm tra thiết bị hoặc driver.")
    return None, -1

# Kiểm tra số lượng ảnh tối thiểu
def check_min_images(count):
    if count < config.MIN_IMAGES_PER_PERSON:
        print(f"Cảnh báo: Chỉ thu thập được {count} ảnh. Nên thu thập ít nhất {config.MIN_IMAGES_PER_PERSON} ảnh.")
    else:
        print(f"Thu thập {count} ảnh, đủ yêu cầu.")

# Chụp ảnh thủ công từ webcam
def capture_images(name, base_folder=config.RAW_IMAGES_DIR, max_images=config.MAX_IMAGES_PER_PERSON):
    if not is_valid_name(name):
        print("Tên không hợp lệ! Chỉ dùng chữ cái, số và dấu cách.")
        return
    
    name = name.strip().replace(" ", "_")
    person_folder = os.path.join(base_folder, name)
    create_folder(person_folder)
    
    if not os.access(base_folder, os.W_OK):
        print(f"Lỗi: Không có quyền ghi vào thư mục {base_folder}")
        return
    
    try:
        yolo = ModelManager.get_yolo()
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLO: {e}")
        return
    
    cap, index = open_webcam()
    if cap is None:
        return
    print(f"Sử dụng webcam index: {index}")
    
    print(f"Đang chụp ảnh cho {name}. Nhấn 'Space' để chụp, 'ESC' để dừng.")
    count = 0
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, config.FRAME_SIZE)
        
        results = yolo(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        face_detected = False
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, classes):
            if int(cls) == 0 and conf >= config.YOLO_CONF_THRESHOLD:
                face_width = x2 - x1
                face_height = y2 - y1
                if face_width >= config.MIN_FACE_SIZE and face_height >= config.MIN_FACE_SIZE:
                    face_detected = True
                    break
        
        status = "Face detected" if face_detected else "No face detected"
        cv2.putText(frame, f"Images: {count}/{max_images} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
        cv2.imshow(f"Chụp ảnh cho {name}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if face_detected:
                if is_good_quality(frame) and frame.shape[0] >= config.MIN_IMAGE_SIZE[0] and frame.shape[1] >= config.MIN_IMAGE_SIZE[1]:
                    count += 1
                    img_name = os.path.join(person_folder, f"{name}_{count}.jpg")
                    try:
                        cv2.imwrite(img_name, frame)
                        print(f"Đã lưu ảnh: {img_name}")
                    except Exception as e:
                        print(f"Lỗi khi lưu ảnh {img_name}: {e}")
                else:
                    print("Ảnh chất lượng thấp, kích thước nhỏ, hoặc khuôn mặt quá nhỏ, thử lại.")
            else:
                print("Không phát hiện khuôn mặt, thử lại.")
        
        if key == 27:
            break
    
    check_min_images(count)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất chụp ảnh.")

if __name__ == "__main__":
    name = input("Nhập tên của người (chỉ dùng chữ cái, số và dấu cách): ")
    capture_images(name)