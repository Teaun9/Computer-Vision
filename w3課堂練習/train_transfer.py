import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# 1. Main Training Setup
# =====================================================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transfer] Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 16
    LR = 0.001
    NUM_EPOCHS = 70
    
    # Data Transformations (RGB, 224x224 for ResNet)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    DATA_DIR = 'D:\Computer-Vision\w2課堂練習\casting_dataset'
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =====================================================================
    # 2. Model Setup (Transfer Learning)
    # =====================================================================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze convolutional layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier with 1 Node for Binary Classification (Unified architecture)
    num_ftrs = model.fc.in_features

    # Replace the old 1000-class with a new 1-node for binary classification
    model.fc = nn.Linear(in_features=num_ftrs, out_features=1)
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    # =====================================================================
    # 3. Training Loop (Standardized)
    # =====================================================================
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.view(-1, 1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.3).float()
            train_correct += (preds == labels).sum().item()

        # --- Validation ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.view(-1, 1).float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()

        # Calculate Metrics
        t_loss = train_loss / len(train_dataset)
        t_acc = (train_correct / len(train_dataset)) * 100
        v_loss = val_loss / len(val_dataset)
        v_acc = (val_correct / len(val_dataset)) * 100

        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'casting_model_transfer.pth')

    # =====================================================================
    # 4. Plotting
    # =====================================================================
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='x')
    plt.title('Loss (Transfer)')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='x')
    plt.title('Accuracy (Transfer)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('casting_curve_transfer.png', dpi=300)
    print("Saved model and plots for Transfer.")

if __name__ == "__main__":
    main()