import os
import torch
from ultralytics import YOLO
from thop import profile
import copy

# ==========================================
# 1. Configure model and dataset paths
# ==========================================
# Path to the best weights from your training run
model_path = os.path.join("runs", "detect", "runs", "object_detection", "yolov8n_exp15", "weights", "best.pt")
data_dir = "object-detection-dataset"  # Name of the split dataset directory
target_split = 'test'

# Validation requires the data.yaml to locate ground-truth labels
yaml_path = f"{data_dir}.yaml"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"❌ Config file not found: {yaml_path}")

# ==========================================
# 2. Detect available hardware
# ==========================================
device_to_use = '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🚀 Starting evaluation (device: {device_to_use})...")

# ==========================================
# 3. Load model
# ==========================================
model = YOLO(model_path)

# ==========================================
# 4. Compute FLOPs and parameter count
# ==========================================
dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)
flop_model = copy.deepcopy(model.model).float().cpu()
flops, params = profile(flop_model, inputs=(dummy_input,), verbose=False)
del flop_model  # free memory

print("\n" + "="*40)
print("🔢 Model Complexity")
print("="*40)
print(f"⚙️  FLOPs:      {flops / 1e9:.2f} GFLOPs")
print(f"🧩 Parameters: {params / 1e6:.2f} M")
print("="*40)

# ==========================================
# 5. Run validation
# ==========================================
print(f"\n📊 Evaluating on test split — please wait...")

metrics = model.val(
    data=yaml_path,
    split=target_split,
    device=device_to_use,
    name=target_split,
    plots=True,
    verbose=False,
    workers=0,
    batch=8
)

# ==========================================
# 6. Extract and display key performance metrics
# ==========================================
precision = metrics.box.mp     # Mean Precision
recall    = metrics.box.mr     # Mean Recall
map50     = metrics.box.map50  # mAP at IoU=0.50
map95     = metrics.box.map    # mAP at IoU=0.50:0.95

print("\n" + "="*40)
print("📋 Test Set Results")
print("="*40)
print(f"🎯 Precision (mean):  {precision:.4f}")
print(f"🧲 Recall (mean):     {recall:.4f}")
print(f"📈 mAP@50:            {map50:.4f}  (lenient threshold)")
print(f"📉 mAP@50-95:         {map95:.4f}  (strict threshold)")
print("="*40)

print("\n💡 Tip: Detailed plots (confusion matrix, F1 curve, etc.) have been saved to 'runs/detect/val/'")