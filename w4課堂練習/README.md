# w4課堂練習
## key feature
- `BATCH_SIZE = 32`
- `LR = 0.001`
-`NUM_EPOCHS = 30`
## architecture design
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
## Successful attempts
1. Install `opencv-python` and `numpy`.
2. Set the input and output folder paths in `CV.py`.
3. Run:
   ```bash
   python CV.py

## Failed attempts