# TSAI ERA V1 A7: Delta-engineering MNIST CNN
> Step-by-step training of a minimal, fully convolutional neural network model on the MNIST dataset
- MNIST: Modified National Institute of Standards and Technology dataset ([Papers With Code](https://paperswithcode.com/dataset/mnist))
- CNN: Convolutional Neural Network model ([Stanford cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks))

## Objectives
Acheive all of the followings using modular code organization:
- Test accuracy >= 99.4 % (consistently shown in the last 2 epochs)
- Number of parameters < 8,000
- Number of epochs <= 15

## Experiments
In order to reach our goals, we iteratively improve one element at a time:
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_01_setup.ipynb) Setup: application layout
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_02_skeleton.ipynb) Basic skeleton: fully convolutional neural network
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_03_dwsc_small.ipynb) Model size decrease: [depthwise separable convolution](https://www.youtube.com/watch?v=vVaRhZXovbw)
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_04_dwsc_big.ipynb) Model size increase: add more parameters
5. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_05_augmentation.ipynb) Data augmentation: image transforms
6. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a7/blob/main/nbs/S7_06_lr.ipynb) Learning rate scheduler: [one cycle policy](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)

## Installation
If you want to use our models, dataloaders, training engine, and other utilities, please run the following command:
```console
git clone https://github.com/woncoh1/era1a7.git
```
And then import the modules in Python:
```python
from era1a7 import data, models, engine, utils
```

## Results summary
- Test accuracy
    - Last: 99.46 % 
    - Best: 99.48 %
- Number of Parameters: 4,392
- Number of Epochs: 15

## Training results
- Best train accuracy: 99.19 %
- Best test accuracy: 99.48 %

![image](https://github.com/woncoh1/era1a7/assets/12987758/aa49b8a3-c169-442e-8787-eba67e867064)

## Receptive field
- r: receptive field size
- n: channel size
- j: jump
- k: kernel size
- s: stride
- p: padding
- conv: convolution layer
- tran: transition layer

| layer | r_in | n_in | j_in | k | s | p | r_out | n_out | j_out |
|-------|------|------|------|---|---|---|-------|-------|-------|
| conv1 |    1 |   28 |    1 | 3 | 1 | 0 |     3 |    26 |     1 |
| conv2 |    3 |   26 |    1 | 3 | 1 | 0 |     5 |    24 |     1 |
| conv3 |    5 |   24 |    1 | 3 | 1 | 0 |     7 |    22 |     1 |
| tran1 |    7 |   22 |    1 | 2 | 2 | 0 |     8 |    11 |     2 |
| conv4 |    8 |   11 |    2 | 3 | 1 | 0 |    12 |     9 |     2 |
| conv5 |   12 |    9 |    2 | 3 | 1 | 0 |    16 |     7 |     2 |
| conv6 |   16 |    7 |    2 | 3 | 1 | 0 |    20 |     5 |     2 |
| tran2 |   20 |    5 |    2 | 5 | 1 | 0 |    28 |     1 |     2 |

## Model summary
`torchinfo.summary` of the final model architecture:
```
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
Model3 (Model3)                          [128, 10]                 --
├─Sequential (conv1)                     [128, 16, 26, 26]         --
│    └─Conv2d (0)                        [128, 16, 26, 26]         144
│    └─ReLU (1)                          [128, 16, 26, 26]         --
│    └─BatchNorm2d (2)                   [128, 16, 26, 26]         32
│    └─Dropout2d (3)                     [128, 16, 26, 26]         --
├─Sequential (conv2)                     [128, 16, 24, 24]         --
│    └─Conv2d (0)                        [128, 16, 24, 24]         144
│    └─ReLU (1)                          [128, 16, 24, 24]         --
│    └─BatchNorm2d (2)                   [128, 16, 24, 24]         32
│    └─Dropout2d (3)                     [128, 16, 24, 24]         --
│    └─Conv2d (4)                        [128, 16, 24, 24]         256
│    └─ReLU (5)                          [128, 16, 24, 24]         --
│    └─BatchNorm2d (6)                   [128, 16, 24, 24]         32
│    └─Dropout2d (7)                     [128, 16, 24, 24]         --
├─Sequential (conv3)                     [128, 32, 22, 22]         --
│    └─Conv2d (0)                        [128, 16, 22, 22]         144
│    └─ReLU (1)                          [128, 16, 22, 22]         --
│    └─BatchNorm2d (2)                   [128, 16, 22, 22]         32
│    └─Dropout2d (3)                     [128, 16, 22, 22]         --
│    └─Conv2d (4)                        [128, 32, 22, 22]         512
│    └─ReLU (5)                          [128, 32, 22, 22]         --
│    └─BatchNorm2d (6)                   [128, 32, 22, 22]         64
│    └─Dropout2d (7)                     [128, 32, 22, 22]         --
├─Sequential (tran1)                     [128, 16, 11, 11]         --
│    └─MaxPool2d (0)                     [128, 32, 11, 11]         --
│    └─Conv2d (1)                        [128, 16, 11, 11]         512
├─Sequential (conv4)                     [128, 16, 9, 9]           --
│    └─Conv2d (0)                        [128, 16, 9, 9]           144
│    └─ReLU (1)                          [128, 16, 9, 9]           --
│    └─BatchNorm2d (2)                   [128, 16, 9, 9]           32
│    └─Dropout2d (3)                     [128, 16, 9, 9]           --
│    └─Conv2d (4)                        [128, 16, 9, 9]           256
│    └─ReLU (5)                          [128, 16, 9, 9]           --
│    └─BatchNorm2d (6)                   [128, 16, 9, 9]           32
│    └─Dropout2d (7)                     [128, 16, 9, 9]           --
├─Sequential (conv5)                     [128, 24, 7, 7]           --
│    └─Conv2d (0)                        [128, 16, 7, 7]           144
│    └─ReLU (1)                          [128, 16, 7, 7]           --
│    └─BatchNorm2d (2)                   [128, 16, 7, 7]           32
│    └─Dropout2d (3)                     [128, 16, 7, 7]           --
│    └─Conv2d (4)                        [128, 24, 7, 7]           384
│    └─ReLU (5)                          [128, 24, 7, 7]           --
│    └─BatchNorm2d (6)                   [128, 24, 7, 7]           48
│    └─Dropout2d (7)                     [128, 24, 7, 7]           --
├─Sequential (conv6)                     [128, 32, 5, 5]           --
│    └─Conv2d (0)                        [128, 24, 5, 5]           216
│    └─ReLU (1)                          [128, 24, 5, 5]           --
│    └─BatchNorm2d (2)                   [128, 24, 5, 5]           48
│    └─Dropout2d (3)                     [128, 24, 5, 5]           --
│    └─Conv2d (4)                        [128, 32, 5, 5]           768
│    └─ReLU (5)                          [128, 32, 5, 5]           --
│    └─BatchNorm2d (6)                   [128, 32, 5, 5]           64
│    └─Dropout2d (7)                     [128, 32, 5, 5]           --
├─Sequential (tran2)                     [128, 10]                 --
│    └─Conv2d (0)                        [128, 10, 5, 5]           320
│    └─AdaptiveAvgPool2d (1)             [128, 10, 1, 1]           --
│    └─Flatten (2)                       [128, 10]                 --
│    └─LogSoftmax (3)                    [128, 10]                 --
==========================================================================================
Total params: 4,392
Trainable params: 4,392
Non-trainable params: 0
Total mult-adds (M): 102.21
==========================================================================================
Input size (MB): 0.40
Forward/backward pass size (MB): 121.91
Params size (MB): 0.02
Estimated Total Size (MB): 122.33
==========================================================================================
```

## Sample images
A set of sample images with their corresponding labels from a batch of the training set:

![image](https://github.com/woncoh1/era1a7/assets/12987758/638e026f-075e-49f4-86f4-d0983295d0e9)

## TODO
- [ ] Demo live predictions
- [ ] Show wrong predictions
