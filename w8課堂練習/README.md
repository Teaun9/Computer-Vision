# w8課堂練習
## training setting（Exp1）

- BATCH_SIZE = `16`
- Image size = `640`
- NUM_EPOCHS = `50`
- patience =  `20`
- workers =  `0`
- seed =  `42`

## Model Complexity（Exp1）

- FLops = `4.10 GFLOPs`
- parameters = `3.01 M`

## Test Set Results（Exp1）

- Precision = `0.8465`
- Recall = `0.7510`
- mAP@50 = `0.8178`
- mAP@50-95 = `0.4631`

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