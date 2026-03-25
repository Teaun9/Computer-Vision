# w4課堂練習
## Key feature

- BATCH_SIZE = `64`
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

## Architecture design
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
- BATCH_SIZE = `64`
- LR = `0.001`
- NUM_EPOCHS = `30`
- Accuracy = `96.11%`
- Macro Precision = `96.27%`
- Macro Recall = `96.11%`
- Macro F1-Score = `96.10%`

## Failed attempts
case1
- BATCH_SIZE = `32`
- LR = `0.001`
- NUM_EPOCHS = `15`
- Accuracy = `95.00%`
- Macro Precision = `95.03%`
- Macro Recall = `95.00%`
- Macro F1-Score = `94.98%`

case2
- BATCH_SIZE = `32`
- LR = `0.001`
- NUM_EPOCHS = `50`
- Accuracy = `95.56%`
- Macro Precision = `95.67%`
- Macro Recall = `95.56%`
- Macro F1-Score = `95.56%`

case3
- BATCH_SIZE = `64`
- LR = `0.001`
- NUM_EPOCHS = `50`
- Accuracy = `95.00%`
- Macro Precision = `95.41%`
- Macro Recall = `95.00%`
- Macro F1-Score = `94.85%`

case4
- BATCH_SIZE = `32`
- LR = `0.001`
- NUM_EPOCHS = `30`
- Accuracy = `93.89%`
- Macro Precision = `94.25%`
- Macro Recall = `93.89%`
- Macro F1-Score = `93.85%`

case5
- BATCH_SIZE = `64`
- LR = `0.001`
- NUM_EPOCHS = `15`
- Accuracy = `93.89%`
- Macro Precision = `93.84%`
- Macro Recall = `93.89%`
- Macro F1-Score = `93.80%`