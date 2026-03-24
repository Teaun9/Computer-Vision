import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. Configuration (The Master Switch)
# =====================================================================
# Set this to 'scratch' or 'transfer' to test the respective model
MODEL_TYPE = 'transfer'  

print(f"--- Preparing to test the '{MODEL_TYPE.upper()}' model ---")

# =====================================================================
# 2. Define the Custom Architecture (Needed if testing 'scratch')
# =====================================================================
class CastingCNN(nn.Module):
    def __init__(self):
        super(CastingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1) # Single output node
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# =====================================================================
# 3. Dynamic Setup based on MODEL_TYPE
# =====================================================================
if MODEL_TYPE == 'scratch':
    # Scratch model uses 128x128 Grayscale
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    model = CastingCNN()
    weights_path = 'casting_model_from_scratch.pth'
    
elif MODEL_TYPE == 'transfer':
    # ResNet18 uses 224x224 RGB
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1) # Single output node
    weights_path = 'casting_model_transfer.pth'

else:
    raise ValueError("MODEL_TYPE must be either 'scratch' or 'transfer'.")

# =====================================================================
# 4. Data & Model Loading
# =====================================================================
DATA_DIR = 'D:\Computer-Vision\w2課堂練習\casting_dataset'
TEST_DIR = os.path.join(DATA_DIR, 'test')

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
class_names = test_dataset.classes

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Cannot find '{weights_path}'. Please train the model first.")

model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
device = torch.device("cpu")
model = model.to(device)

# =====================================================================
# 5. Unified Evaluation Loop
# =====================================================================
model.eval()
all_preds = []
all_labels = []

print("Running inference on test dataset...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        # Reshape labels to (Batch, 1) to match the single output node
        labels = labels.view(-1, 1).float().to(device)
        
        outputs = model(inputs)
        
        # --- UNIVERSAL LOGIC FOR BOTH MODELS ---
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

# =====================================================================
# 6. Calculate Metrics & Plot
# =====================================================================
test_acc = (sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)) * 100
# pos_label=0 ensures we measure the ability to catch defects ('def_front')
precision = precision_score(all_labels, all_preds, pos_label=1) * 100
recall = recall_score(all_labels, all_preds, pos_label=1) * 100
f1 = f1_score(all_labels, all_preds, pos_label=1) * 100

print(f"\n--- Final Test Results ({MODEL_TYPE.upper()}) ---")
print(f"Accuracy : {test_acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-Score : {f1:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})

title_text = f'Confusion Matrix ({MODEL_TYPE.upper()})\nAcc: {test_acc:.1f}% | Pre: {precision:.1f}% | Rec: {recall:.1f}%'
plt.title(title_text, fontsize=14)
plt.ylabel('Actual Condition (True Label)', fontsize=12)
plt.xlabel('Predicted Condition (Model Output)', fontsize=12)

plt.tight_layout()
save_name = f'casting_confusion_matrix_{MODEL_TYPE}.png'
plt.savefig(save_name, dpi=300)
print(f"Saved confusion matrix as '{save_name}'")