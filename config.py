import os

# Đường dẫn thư mục
BASE_DIR = 'datasets'
RAW_IMAGES_DIR = os.path.join(BASE_DIR, 'raw_images')
COLLECTED_IMAGES_DIR = os.path.join(BASE_DIR, 'collected_images')
AUGMENTED_IMAGES_DIR = os.path.join(BASE_DIR, 'augmented_images')
CROPPED_IMAGES_DIR = os.path.join(BASE_DIR, 'cropped_images')
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test_images')
EMBEDDINGS_DIR = 'embeddings'
MODELS_DIR = 'models'
CHART_DIR = 'charts'

# Tham số cho chụp ảnh
MAX_IMAGES_PER_PERSON = 50
MIN_IMAGES_PER_PERSON = 50
DELAY_AUTO_CAPTURE = 2.0  # giây
FRAME_SIZE = (800, 600)  # Tăng để có khung hình lớn hơn

# Tham số cho augmentation
TARGET_IMAGES_PER_PERSON = 800  # Tăng để tạo nhiều ảnh hơn
TARGET_TEST_IMAGES_PER_PERSON = 300
MIN_FACE_SIZE = 80  # Tăng để yêu cầu khuôn mặt lớn hơn
MIN_IMAGE_SIZE = (50, 50)  # Điều chỉnh tương ứng với MIN_FACE_SIZE
BATCH_SIZE = 32

# Tham số cho cắt khuôn mặt
TARGET_CROPPED_IMAGES = 500  # Tăng để giữ nhiều khuôn mặt hơn
QUALITY_BRIGHTNESS_RANGE = (20, 230)
QUALITY_SHARPNESS_THRESHOLD = 10

# Tham số cho FaceNet
FACENET_IMAGE_SIZE = (160, 160)  # Tăng để cải thiện chất lượng embedding
FACENET_MEAN = [0.5, 0.5, 0.5]
FACENET_STD = [0.5, 0.5, 0.5]

# Tham số cho SVM
SVM_PROB_THRESHOLD = 0.7
SVM_COSINE_THRESHOLD = 0.8
SVM_COSINE_THRESHOLD_UNKNOWN = 0.65

# Tham số cho YOLO
YOLO_CONF_THRESHOLD = 0.6

# Đường dẫn file mô hình và embedding
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov11s-face.pt')
FACENET_FINETUNED_PATH = os.path.join(MODELS_DIR, 'facenet_classifier.pth')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, 'embeddings.npy')
LABELS_PATH = os.path.join(EMBEDDINGS_DIR, 'labels.npy')