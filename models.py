import os
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch
import config

class ModelManager:
    _yolo = None
    _facenet = None
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def get_yolo(cls):
        if cls._yolo is None:
            try:
                cls._yolo = YOLO(config.YOLO_MODEL_PATH)
                cls._yolo.conf = config.YOLO_CONF_THRESHOLD
            except Exception as e:
                print(f"Lỗi khi tải mô hình YOLO: {e}")
                exit(1)
        return cls._yolo

    @classmethod
    def get_facenet(cls):
        if cls._facenet is None:
            try:
                cls._facenet = InceptionResnetV1(pretrained='vggface2').eval().to(cls._device)
                if os.path.exists(config.FACENET_FINETUNED_PATH):
                    state_dict = torch.load(config.FACENET_FINETUNED_PATH, map_location=cls._device)
                    cls._facenet.load_state_dict(state_dict)
                    print("Đã tải mô hình FaceNet đã fine-tuned")
                else:
                    print("Cảnh báo: Không tìm thấy mô hình fine-tuned. Sử dụng mô hình pretrained.")
            except Exception as e:
                print(f"Lỗi khi tải mô hình FaceNet: {e}")
                exit(1)
        return cls._facenet, cls._device