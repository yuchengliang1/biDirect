import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# ---------- 导入项目模块 ----------
from data_read import load_eeg_and_meg_data
from TSMMF import HybridTransformer


# ==================================
# 1️⃣ 创建自定义 Dataset
# ==================================
class EEGfNIRSDataset(Dataset):
    def __init__(self, eeg, nirs, labels):
        self.eeg = eeg
        self.nirs = nirs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.nirs[idx], self.labels[idx]


# ==================================
# 2️⃣ 准备数据集与 DataLoader
# ==================================
# 分割训练集与验证集（8:2）
eeg, nirs, labels = load_eeg_and_meg_data()
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels)

train_dataset = EEGfNIRSDataset(eeg[train_idx], nirs[train_idx], labels[train_idx])
val_dataset   = EEGfNIRSDataset(eeg[val_idx], nirs[val_idx], labels[val_idx])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)


# ==================================
# 3️⃣ 定义模型、损失函数、优化器
# ==================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = HybridTransformer(
    depth=[2, 1],         # [self-encoder层数, cross-encoder层数]
    query_size=64,
    key_size=64,
    value_size=64,
    emb_size=64,
    num_heads=4,
    expansion=2,
    conv_dropout=0.2,
    self_dropout=0.2,
    cross_dropout=0.2,
    cls_dropout=0.2,
    num_classes=3,
    device=device
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# ==================================
# 4️⃣ 定义训练和验证函数
# ==================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0

    for eeg_batch, nirs_batch, label_batch in tqdm(loader, desc="Training", leave=False):
        eeg_batch, nirs_batch, label_batch = eeg_batch.to(device), nirs_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        outputs = model(eeg_batch, nirs_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label_batch.size(0)
        pred = outputs.argmax(dim=1)
        total_correct += (pred == label_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    for eeg_batch, nirs_batch, label_batch in tqdm(loader, desc="Validating", leave=False):
        eeg_batch, nirs_batch, label_batch = eeg_batch.to(device), nirs_batch.to(device), label_batch.to(device)
        outputs = model(eeg_batch, nirs_batch)
        loss = criterion(outputs, label_batch)

        total_loss += loss.item() * label_batch.size(0)
        pred = outputs.argmax(dim=1)
        total_correct += (pred == label_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


# ==================================
# 5️⃣ 训练主循环
# ==================================
num_epochs = 300
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    print(f"\n===== Epoch {epoch}/{num_epochs} =====")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f" Val  Loss: {val_loss:.4f},  Val  Acc: {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_TSMMF_model.pth")
        print(f"✅ Saved best model with accuracy {best_val_acc:.4f}")

print("\nTraining Complete ✅")