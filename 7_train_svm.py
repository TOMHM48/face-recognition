import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import label_binarize
import joblib
from collections import Counter
import config

if not os.path.exists(config.EMBEDDINGS_PATH) or not os.path.exists(config.LABELS_PATH):
    print("Lỗi: Không tìm thấy file 'embeddings.npy' hoặc 'labels.npy'.")
    exit()

try:
    embeddings = np.load(config.EMBEDDINGS_PATH)
    labels = np.load(config.LABELS_PATH)
except Exception as e:
    print(f"Lỗi khi tải embeddings/labels: {e}")
    exit()

if embeddings.shape[0] < 2:
    print("Lỗi: Không đủ embeddings để huấn luyện!")
    exit()
if len(np.unique(labels)) < 2:
    print("Lỗi: Cần ít nhất 2 nhãn để huấn luyện!")
    exit()

print("Số lượng mẫu cho mỗi nhãn:")
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"{label}: {count}")
    if count < config.MIN_IMAGES_PER_PERSON:
        print(f"Cảnh báo: Số lượng mẫu cho {label} quá ít ({count}).")
if max(label_counts.values()) / min(label_counts.values()) > 5:
    print("Cảnh báo: Dữ liệu không cân bằng! Tỷ lệ lớn nhất/nhỏ nhất: {:.2f}".format(
        max(label_counts.values()) / min(label_counts.values())))
print(f"Tổng số mẫu: {len(embeddings)}")
print(f"Số lượng lớp: {len(np.unique(labels))}")

def balance_data(embeddings, labels, max_samples_per_label=300):
    balanced_emb = []
    balanced_labels = []
    for label in np.unique(labels):
        mask = labels == label
        indices = np.where(mask)[0]
        if len(indices) > max_samples_per_label:
            indices = np.random.choice(indices, max_samples_per_label, replace=False)
        balanced_emb.append(embeddings[indices])
        balanced_labels.extend([label] * len(indices))
    return np.vstack(balanced_emb), np.array(balanced_labels)

embeddings, labels = balance_data(embeddings, labels)
print(f"Số mẫu sau khi cân bằng: {len(embeddings)}")

unique_labels = np.unique(labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

labels_idx = np.array([label_to_idx[label] for label in labels])

X_temp, X_test, y_temp, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

def add_noise(X, factor):
    return np.clip(X + np.random.normal(0, factor, X.shape), -1, 1)

X_train = add_noise(X_train, 0.05)
X_val = add_noise(X_val, 0.02)

print(f"Số mẫu train: {len(X_train)}")
print(f"Số mẫu validation: {len(X_val)}")
print(f"Số mẫu test: {len(X_test)}")

param_grid = {
    'C': [0.1, 1.0, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.001]
}

clf = SVC(probability=True, class_weight='balanced')

try:
    grid_search = RandomizedSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
    grid_search.fit(X_train, y_train)
except Exception as e:
    print(f"Lỗi khi chạy RandomizedSearchCV: {e}")
    exit()

print("\nTham số tốt nhất:")
print(grid_search.best_params_)
print(f"Độ chính xác tốt nhất (CV): {grid_search.best_score_:.3f}")

best_clf = grid_search.best_estimator_
for name, X, y in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
    try:
        y_pred = best_clf.predict(X)
        y_proba = best_clf.predict_proba(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        y_bin = label_binarize(y, classes=unique_labels)
        auc = roc_auc_score(y_bin, y_proba, multi_class='ovr')
        print(f"\nĐánh giá trên tập {name}:")
        print(f"Độ chính xác: {acc:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC: {auc:.3f}")
        print("Classification Report:")
        print(classification_report(y, y_pred, target_names=unique_labels))
        print(f"Ma trận nhầm lẫn ({name}):")
        cm = confusion_matrix(y, y_pred, labels=unique_labels)
        print(f"  Labels: {unique_labels.tolist()}")
        print(cm)
        print()
    except Exception as e:
        print(f"Lỗi khi đánh giá trên tập {name}: {e}")

try:
    if not os.access(config.MODELS_DIR, os.W_OK):
        raise PermissionError("Không có quyền ghi!")
    joblib.dump(best_clf, config.SVM_MODEL_PATH)
    print(f"\nĐã lưu mô hình vào '{config.SVM_MODEL_PATH}'")
except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")