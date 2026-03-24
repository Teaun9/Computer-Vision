import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# 1. Define the Custom CNN Model
# =====================================================================
class CastingCNN(nn.Module):
    def __init__(self):
        super(CastingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        flattened_size = 64 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1) # 1 Node for Binary
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =====================================================================
# 2. Main Training Setup
# =====================================================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[From Scratch] Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 16
    LR = 0.001
    NUM_EPOCHS = 70
    
    # Data Transformations (Grayscale, 128x128)
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load Datasets
    DATA_DIR = 'D:/Computer-Vision/w2課堂練習/casting_dataset'
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = CastingCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

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
            torch.save(model.state_dict(), 'casting_model_from_scratch.pth')

    # =====================================================================
    # 4. Plotting
    # =====================================================================
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='x')
    plt.title('Loss (From Scratch)')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='x')
    plt.title('Accuracy (From Scratch)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('casting_curve_from_scratch.png', dpi=300)
    print("Saved model and plots for From Scratch.")

if __name__ == "__main__":
    main()