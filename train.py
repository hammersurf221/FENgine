import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import ChessBoardDataset
from ccn_model import CCN
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 📁 Make sure the folder exists
os.makedirs("Confusion_Matrixes", exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved confusion matrix to {filename}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 13

full_dataset = ChessBoardDataset("data/train")
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CCN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images).permute(0, 3, 1, 2)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).permute(0, 3, 1, 2)
            val_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Validation Loss = {val_loss:.4f}, Accuracy = {val_acc:.2%}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "ccn_model.pth")
        print("✅ Saved best model")

    # Save confusion matrix
    epoch_str = f"{epoch+1:02d}"
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=[".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"],
        filename=f"Confusion_Matrixes/confusion_matrix_E{epoch_str}.png"
    )
