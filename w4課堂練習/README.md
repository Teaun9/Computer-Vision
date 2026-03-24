# w4課堂練習
## key feature

- BATCH_SIZE = `32`
- LR = `0.001`
- NUM_EPOCHS = `30`
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`

## Data Preprocessing

- Resize images to `128 x 128`
- Convert images to grayscale
- Convert images to tensor
- Normalize with:
  - mean = 0.5
  - std = 0.5

## architecture design
The network consists of:
- an initial convolution layer with 16 output channels
- a residual block at 16 channels
- max-pooling
- a second convolution layer with 32 output channels
- a residual block at 32 channels
- max-pooling
- a third convolution layer with 64 output channels
- a residual block at 64 channels
- max-pooling
- a fully connected classifier with one hidden layer and dropout

## Successful attempts
1. Install `opencv-python` and `numpy`.
2. Set the input and output folder paths in `CV.py`.
3. Run:
   ```bash
   python CV.py

## Failed attempts