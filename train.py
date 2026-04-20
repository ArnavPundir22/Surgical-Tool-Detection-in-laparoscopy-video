import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# ================= CONFIG =================
DATA_DIR = "cholec-tinytools"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= DATA =================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)

# ================= MODEL =================
model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= LOG STORAGE =================
log_data = {
    "config": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "model": "resnet50"
    },
    "epochs": []
}

train_losses, val_accuracies, train_accuracies = [], [], []

# ================= TRAIN FUNCTION =================
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_ds)
    return total_loss, acc

# ================= VALIDATION =================
def validate():
    model.eval()
    correct = 0

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / len(val_ds)
    return acc, all_preds, all_labels

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):
    loss, train_acc = train_one_epoch()
    val_acc, preds, labels = validate()

    train_losses.append(loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    epoch_log = {
        "epoch": epoch + 1,
        "loss": loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    }

    log_data["epochs"].append(epoch_log)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "resnet50_tools.pth"))

# ================= SAVE JSON LOG =================
with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(log_data, f, indent=4)

print("Logs saved!")

# ================= PLOTS =================

# Loss Curve
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

# Accuracy Curve
plt.figure()
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(labels, preds)
df_cm = pd.DataFrame(cm, index=train_ds.classes, columns=train_ds.classes)

plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("Graphs saved in outputs/")
