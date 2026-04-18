import os
import sys
import yaml
import torch
from ultralytics import YOLO

# ==========================================
# 0. Cross-platform logger (simulates tee command)
# ==========================================
class DualLogger:
    """Intercepts all print calls and writes output to both the terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout  # Keep a reference to the original stdout
        self.log = open(filepath, "w", encoding="utf-8")  # Open the target log file

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log.write(message)       # Write to file
        self.log.flush()              # Flush immediately to avoid data loss on crash

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Activate the logger — all output will be mirrored to training_log.txt
sys.stdout = DualLogger("training_log.txt")
sys.stderr = sys.stdout  # Also capture stderr so error messages are logged

print("Logger active — all output will be saved to training_log.txt\n")

# ==========================================
# 1. Read classes.txt and parse class names
# ==========================================
dataset_dir = "object-detection-dataset"  # Name of the split dataset directory

# Look for classes.txt inside the labels/ folder (matches the structure from the previous step)
classes_file = os.path.join(dataset_dir, "labels", "classes.txt")
if not os.path.exists(classes_file):
    raise FileNotFoundError(f"❌ Error: {classes_file} not found. Please check that the file exists.")

# Read and parse the file, stripping whitespace and skipping empty lines
with open(classes_file, "r", encoding="utf-8") as f:
    class_list = [line.strip() for line in f.readlines() if line.strip()]

# Convert the list to the dictionary format expected by YOLO
names_dict = {i: name for i, name in enumerate(class_list)}
print(f"✅ Classes loaded: {names_dict}")

# ==========================================
# 2. Auto-generate the data.yaml config file required by YOLO
# ==========================================

yaml_content = {
    "path": os.path.abspath(dataset_dir),  # Absolute path avoids working-directory issues
    "train": "images/train",               # Training images
    "val": "images/val",                   # Validation images
    "test": "images/test",                 # Test images
    "names": names_dict
}

yaml_path = f"{dataset_dir}.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_content, f, sort_keys=False)
print(f"✅ Dataset config written to: {yaml_path}")

# ==========================================
# 3. Load model and configure training
# ==========================================
print("⏳ Loading YOLOv8n model...")
# The latest yolov8n.pt weights will be downloaded automatically if not cached
model = YOLO('yolov8n.pt')

# ==========================================
# 4. Detect available hardware and start training
# ==========================================
# Select the best available compute device
device_to_use = '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"🚀 Starting training on device: {device_to_use}")

results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    device=device_to_use,  # Dynamically set based on detected hardware
    batch=16,               # Adjust batch size to match available memory
    patience=20,
    workers=0,
    seed=42,
    project="runs/object_detection",
    name="yolov8n_exp1"
)

print("\n🎉 Training complete!")