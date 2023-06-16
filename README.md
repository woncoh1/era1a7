# TSAI ERA V1 A7: Delta-engineering MNIST CNN
> Step-by-step training of a minimal, fully convolutional neural network model on the MNIST dataset
- MNIST: Modified National Institute of Standards and Technology dataset ([Papers With Code](https://paperswithcode.com/dataset/mnist))
- CNN: Convolutional Neural Network model ([Stanford cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks))

## Objective
Acheive all of the followings using modular code organization:
- Test accuracy >= 99.4 % (consistently shown in the last 2 epochs)
- Number of parameters < 8,000
- Number of epochs <= 15

## Model summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
         Dropout2d-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]             144
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
         Dropout2d-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 16, 24, 24]             256
             ReLU-10           [-1, 16, 24, 24]               0
      BatchNorm2d-11           [-1, 16, 24, 24]              32
        Dropout2d-12           [-1, 16, 24, 24]               0
           Conv2d-13           [-1, 16, 22, 22]             144
             ReLU-14           [-1, 16, 22, 22]               0
      BatchNorm2d-15           [-1, 16, 22, 22]              32
        Dropout2d-16           [-1, 16, 22, 22]               0
           Conv2d-17           [-1, 32, 22, 22]             512
             ReLU-18           [-1, 32, 22, 22]               0
      BatchNorm2d-19           [-1, 32, 22, 22]              64
        Dropout2d-20           [-1, 32, 22, 22]               0
        MaxPool2d-21           [-1, 32, 11, 11]               0
           Conv2d-22           [-1, 16, 11, 11]             512
           Conv2d-23             [-1, 16, 9, 9]             144
             ReLU-24             [-1, 16, 9, 9]               0
      BatchNorm2d-25             [-1, 16, 9, 9]              32
        Dropout2d-26             [-1, 16, 9, 9]               0
           Conv2d-27             [-1, 16, 9, 9]             256
             ReLU-28             [-1, 16, 9, 9]               0
      BatchNorm2d-29             [-1, 16, 9, 9]              32
        Dropout2d-30             [-1, 16, 9, 9]               0
           Conv2d-31             [-1, 16, 7, 7]             144
             ReLU-32             [-1, 16, 7, 7]               0
      BatchNorm2d-33             [-1, 16, 7, 7]              32
        Dropout2d-34             [-1, 16, 7, 7]               0
           Conv2d-35             [-1, 24, 7, 7]             384
             ReLU-36             [-1, 24, 7, 7]               0
      BatchNorm2d-37             [-1, 24, 7, 7]              48
        Dropout2d-38             [-1, 24, 7, 7]               0
           Conv2d-39             [-1, 24, 5, 5]             216
             ReLU-40             [-1, 24, 5, 5]               0
      BatchNorm2d-41             [-1, 24, 5, 5]              48
        Dropout2d-42             [-1, 24, 5, 5]               0
           Conv2d-43             [-1, 32, 5, 5]             768
             ReLU-44             [-1, 32, 5, 5]               0
      BatchNorm2d-45             [-1, 32, 5, 5]              64
        Dropout2d-46             [-1, 32, 5, 5]               0
           Conv2d-47             [-1, 10, 5, 5]             320
AdaptiveAvgPool2d-48             [-1, 10, 1, 1]               0
          Flatten-49                   [-1, 10]               0
       LogSoftmax-50                   [-1, 10]               0
================================================================
Total params: 4,392
Trainable params: 4,392
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.83
Params size (MB): 0.02
Estimated Total Size (MB): 1.85
----------------------------------------------------------------
```

## Training results
- Best train accuracy: 99.19 %
- Best test accuracy: 99.48 %
![image](https://github.com/woncoh1/era1a7/assets/12987758/aa49b8a3-c169-442e-8787-eba67e867064)

## TODO
- [ ] Demo live predictions
- [ ] Show wrong predictions
