# w2課堂練習
## pipeline steps
- 從輸入資料夾讀取所有圖片
- 進行彩色影像去雜訊
- 將圖片進行局部對比增強
- 使用 unsharp mask 提升邊緣清晰度
- 輸出原圖與處理後圖片
## parameter choices
1. 去雜訊：
```
cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
```
  - `h = 15`
  - `hColor = 15`
  - `templateWindowSize = 7`
  - `searchWindowSize = 21`

2. 對比增強：
```
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
  - `clipLimit = 2.0`
  - `tileGridSize = (8, 8)`

3. 銳化：
```
unsharp_mask(img, sigma=0.8, strength=0.5)
```
  - `sigma = 0.8`
  - `strength = 0.5`
## run code
1. Install `opencv-python` and `numpy`.
2. Set the input and output folder paths in `CV.py`.
3. Run:
   ```bash
   python CV.py
