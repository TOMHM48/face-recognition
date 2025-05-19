import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import config

if not os.path.exists(config.EMBEDDINGS_PATH) or not os.path.exists(config.LABELS_PATH):
    raise FileNotFoundError("Không tìm thấy file 'embeddings.npy' hoặc 'labels.npy'.")

try:
    embeddings = np.load(config.EMBEDDINGS_PATH)
    labels = np.load(config.LABELS_PATH)
except Exception as e:
    print(f"Lỗi khi tải embeddings/labels: {e}")
    exit()

if embeddings.shape[0] < 2:
    print("Lỗi: Không đủ embeddings để trực quan hóa (cần ít nhất 2 mẫu)!")
    exit()
if embeddings.shape[1] != 512:
    print(f"Lỗi: Kích thước embeddings không đúng ({embeddings.shape[1]} thay vì 512)!")
    exit()
if len(np.unique(labels)) < 2:
    print("Cảnh báo: Chỉ có 1 nhãn, biểu đồ sẽ không có ý nghĩa phân loại!")

print(f"Tổng số embeddings: {embeddings.shape[0]}, Số nhãn: {len(np.unique(labels))}")

# Lấy mẫu ngẫu nhiên nếu quá nhiều embedding
max_samples = 10000
if embeddings.shape[0] > max_samples:
    indices = np.random.choice(embeddings.shape[0], max_samples, replace=False)
    embeddings = embeddings[indices]
    labels = labels[indices]
    print(f"Đã lấy mẫu ngẫu nhiên {max_samples} embeddings để tối ưu t-SNE")

perplexity = min(30, max(5, embeddings.shape[0] // 5))
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
except Exception as e:
    print(f"Lỗi khi chạy t-SNE: {e}")
    exit()

plt.figure(figsize=(14, 10))
unique_labels = np.unique(labels)
colors = sns.color_palette("husl", len(unique_labels))

for i, label in enumerate(unique_labels):
    mask = labels == label
    count = np.sum(mask)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                label=f"{label} ({count} samples)", color=colors[i], alpha=0.6, s=100)

plt.title("Scatter Plot of Face Embeddings (t-SNE)", fontsize=14)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Labels")
plt.grid(True)
plt.tight_layout()

os.makedirs(config.CHART_DIR, exist_ok=True)
try:
    if not os.access(config.CHART_DIR, os.W_OK):
        raise PermissionError(f"Không có quyền ghi vào thư mục {config.CHART_DIR}/!")
    scatter_path = os.path.join(config.CHART_DIR, "embedding_scatter.png")
    plt.savefig(scatter_path, bbox_inches='tight')
    print(f"Đã lưu biểu đồ vào '{scatter_path}'")
except Exception as e:
    print(f"Lỗi khi lưu biểu đồ: {e}")
plt.close()

print("\nĐộ phân tán của các điểm trong mỗi nhãn (khoảng cách trung bình đến tâm cụm):")
for label in unique_labels:
    mask = labels == label
    cluster = embeddings_2d[mask]
    if len(cluster) > 1:
        centroid = np.mean(cluster, axis=0)
        distances = np.sqrt(((cluster - centroid) ** 2).sum(axis=1))
        mean_distance = np.mean(distances)
        print(f"{label}: Mean distance to centroid = {mean_distance:.3f}")