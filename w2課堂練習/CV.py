import cv2
import numpy as np
import os

input_dir = r"D:\Computer Vision_class\w1"
output_dir = r"D:\output_final"
os.makedirs(output_dir, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def read_image_unicode(path):
    file_bytes = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def save_image_unicode(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        encoded_img.tofile(path)
    else:
        print(f"儲存失敗: {path}")


def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)

def enhance_contrast_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def unsharp_mask(img, sigma=0.8, strength=0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

def process_blurry_image(img):
    result = denoise_image(img)
    result = enhance_contrast_clahe(result)
    result = unsharp_mask(result, sigma=0.8, strength=0.5)
    return result

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

if not image_files:
    print("找不到圖片，請確認 input_dir 路徑是否正確。")
    exit()

print(f"找到 {len(image_files)} 張圖片，開始處理模糊照片...")

for file_name in image_files:
    img_path = os.path.join(input_dir, file_name)
    img = read_image_unicode(img_path)

    if img is None:
        print(f"無法讀取: {file_name}")
        continue

    base_name, ext = os.path.splitext(file_name)

    original = img.copy()
    result = process_blurry_image(original)

    save_image_unicode(os.path.join(output_dir, f"{base_name}_before{ext}"), original)
    save_image_unicode(os.path.join(output_dir, f"{base_name}_after{ext}"), result)

    print(f"處理完成: {file_name}")

print("全部完成！")
print(f"輸出資料夾: {output_dir}")