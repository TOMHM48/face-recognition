import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from models import ModelManager
import config

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(self.labels) < 2:
            raise ValueError(f"Lỗi: Cần ít nhất 2 nhãn, hiện chỉ có {len(self.labels)} nhãn!")
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.image_paths = []
        self.image_labels = []
        for label in self.labels:
            label_dir = os.path.join(data_dir, label)
            images = [img for img in os.listdir(label_dir) if img.lower().endswith(('.jpg', '.png'))]
            for img_name in images:
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.image_labels.append(self.label_to_idx[label])
        if len(self.image_paths) < 100:
            raise ValueError(f"Lỗi: Không đủ ảnh, chỉ có {len(self.image_paths)} ảnh trong {data_dir}!")
        label_counts = {}
        for label in self.labels:
            label_counts[label] = sum(1 for l in self.image_labels if self.label_to_idx[label] == l)
            print(f"{label}: {label_counts[label]} ảnh")
            if label_counts[label] < config.MIN_IMAGES_PER_PERSON:
                print(f"Cảnh báo: Số lượng ảnh cho {label} quá ít ({label_counts[label]}).")
        print(f"Tổng số ảnh: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            return None, None
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(config.FACENET_IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.FACENET_MEAN, std=config.FACENET_STD),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

try:
    dataset = FaceDataset(data_dir=config.CROPPED_IMAGES_DIR, transform=transform)
except Exception as e:
    print(f"Lỗi khi tải dataset: {e}")
    exit()

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"Số mẫu train: {len(train_dataset)}")
print(f"Số mẫu validation: {len(val_dataset)}")
print(f"Số mẫu test: {len(test_dataset)}")

facenet, device = ModelManager.get_facenet()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, classes=10):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FaceNetClassifier(nn.Module):
    def __init__(self, facenet, num_classes):
        super(FaceNetClassifier, self).__init__()
        self.facenet = facenet
        self.bn = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        embedding = torch.nn.functional.normalize(self.facenet(x))
        x = self.bn(embedding)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits, embedding

model = FaceNetClassifier(facenet, num_classes=len(dataset.labels)).to(device)

total_layers = len(list(model.named_parameters()))
freeze_until = int(total_layers * 0.3)
for idx, (name, param) in enumerate(model.named_parameters()):
    if idx < freeze_until and 'fc' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

criterion = LabelSmoothingLoss(smoothing=0.1, classes=len(dataset.labels))
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-6,
    weight_decay=0.05
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

num_epochs = 100
patience = 10
best_val_accuracy = 0.0
patience_counter = 0

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        if images is None or labels is None:
            continue
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total if train_total > 0 else 0
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            if images is None or labels is None:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total if val_total > 0 else 0
    val_losses.append(val_loss / len(val_loader))
    scheduler.step(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        try:
            torch.save(model.facenet.state_dict(), config.FACENET_FINETUNED_PATH)
            print(f"Saved fine-tuned FaceNet with Val Acc: {best_val_accuracy:.4f}")
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

model.eval()
test_correct = 0
test_total = 0
test_predictions = []
test_labels = []
test_predictions_proba = []
with torch.no_grad():
    for images, labels in test_loader:
        if images is None or labels is None:
            continue
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_predictions_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

test_accuracy = test_correct / test_total if test_total > 0 else 0
test_f1 = f1_score(test_labels, test_predictions, average='weighted')
test_labels_bin = label_binarize(test_labels, classes=range(len(dataset.labels)))
test_auc = roc_auc_score(test_labels_bin, test_predictions_proba, multi_class='ovr')

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")

if train_accuracy - test_accuracy > 0.1:
    print("Cảnh báo: Có thể đang overfitting! Thu thập thêm dữ liệu hoặc tăng regularisation.")
if test_accuracy < 0.7:
    print("Cảnh báo: Độ chính xác trên tập test thấp! Kiểm tra dữ liệu hoặc điều chỉnh hyperparameters.")

print("Classification Report:")
target_names = [dataset.idx_to_label[i] for i in range(len(dataset.labels))]
print(classification_report(test_labels, test_predictions, target_names=target_names))

print("Confusion Matrix:")
cm = confusion_matrix(test_labels, test_predictions)
print("  Labels:", target_names)
print(cm)

print("Finetuning completed!")