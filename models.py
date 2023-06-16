import torch
import torch.nn as nn


def conv( # Convolution layer
    i:int, # in_channels
    o:int, # out_channels
    p:int=0, # padding
    d:float=0, # dropout rate
) -> nn.Sequential:
    return nn.Sequential(
        # 3x3 convolution to extract features
        nn.Conv2d(
            in_channels=i,
            out_channels=o,
            kernel_size=3,
            stride=1,
            padding=p,
            padding_mode='replicate',
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=o),
        nn.Dropout2d(p=d),
    )


def dwsc( # Depthwise separable convolution layer
    i:int, # in_channels
    o:int, # out_channels
    p:int=0, # padding
    d:float=0, # dropout rate
) -> nn.Sequential:
    # https://www.youtube.com/watch?v=vVaRhZXovbw
    # https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py#L15-L26
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L42-L53
    return nn.Sequential(
        # Depth-wise convolution
        nn.Conv2d(
            in_channels=i,
            out_channels=i,
            kernel_size=3,
            stride=1,
            padding=p,
            padding_mode='replicate',
            groups=i,
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=i),
        nn.Dropout2d(p=d),
        # Point-wise convolution
        nn.Conv2d(
            in_channels=i,
            out_channels=o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=o),
        nn.Dropout2d(p=d),
    )


def tran( # Transition layer = MaxPooling + 1x1 convolution
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    return nn.Sequential(
        # MaxPooling to reduce the channel size
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 1x1 convolution to reduce the number of channels
        nn.Conv2d(
            in_channels=i,
            out_channels=o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
    )


def last( # Prediction layer = GAP + softmax
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    return nn.Sequential(
        # [-1, i, s, s]
        nn.Conv2d(
            in_channels=i,
            out_channels=o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        # [-1, o, s, s]
        nn.AdaptiveAvgPool2d(output_size=1),
        # [-1, o, 1, 1]
        nn.Flatten(),
        # [-1, o]
        nn.LogSoftmax(dim=1), # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
        # [-1, o]
    )


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(1, 8)
        self.conv2 = conv(8, 16)
        self.tran1 = tran(16, 8)
        self.conv3 = conv(8, 16)
        self.conv4 = conv(16, 16)
        self.tran2 = tran(16, 8)
        self.conv5 = conv(8, 16, p=1)
        self.conv6 = conv(16, 16, p=1)
        self.tran3 = last(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return toolz.pipe(x,
            # Block 1: edges and gradients
            self.conv1,
            self.conv2,
            self.tran1,
            # Block 2: textures and patterns
            self.conv3,
            self.conv4,
            self.tran2,
             # Block 3: objects
            self.conv5,
            self.conv6,
            self.tran3,
        )


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(1, 8) # n=26, r=3, j=1
        self.conv2 = conv(8, 16) # n=24, r=5, j=1
        self.conv3 = conv(16, 16) # n=22, r=7, j=1
        self.tran1 = tran(16, 8) # n=11, r=8, j=2
        self.conv4 = conv(8, 16) # n=9, r=12, j=2
        self.conv5 = conv(16, 16) # n=7, r=16, j=2
        self.conv6 = conv(16, 16) # n=5, r=20, j=2
        self.tran2 = last(16, 10) # n=1, r=28, j=2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return toolz.pipe(x,
            # Block 1: edges and gradients
            self.conv1,
            self.conv2,
            self.conv3,
            self.tran1,
            # Block 2: objects
            self.conv4,
            self.conv5,
            self.conv6,
            self.tran2,
        )


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = dwsc(1, 8) # n=26, r=3, j=1
        self.conv2 = dwsc(8, 16) # n=24, r=5, j=1
        self.conv3 = dwsc(16, 16) # n=22, r=7, j=1
        self.tran1 = tran(16, 8) # n=11, r=8, j=2
        self.conv4 = dwsc(8, 16) # n=9, r=12, j=2
        self.conv5 = dwsc(16, 16) # n=7, r=16, j=2
        self.conv6 = dwsc(16, 16) # n=5, r=20, j=2
        self.tran2 = last(16, 10) # n=1, r=28, j=2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return toolz.pipe(x,
            # Block 1: edges and gradients
            self.conv1,
            self.conv2,
            self.conv3,
            self.tran1,
            # Block 2: objects
            self.conv4,
            self.conv5,
            self.conv6,
            self.tran2,
        )


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(1, 16) # n=26, r=3, j=1
        self.conv2 = dwsc(16, 16) # n=24, r=5, j=1
        self.conv3 = dwsc(16, 32) # n=22, r=7, j=1
        self.tran1 = tran(32, 16) # n=11, r=8, j=2
        self.conv4 = dwsc(16, 16) # n=9, r=12, j=2
        self.conv5 = dwsc(16, 24) # n=7, r=16, j=2
        self.conv6 = dwsc(24, 32) # n=5, r=20, j=2
        self.tran2 = last(32, 10) # n=1, r=28, j=2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return toolz.pipe(x,
            # Block 1: edges and gradients
            self.conv1,
            self.conv2,
            self.conv3,
            self.tran1,
            # Block 2: objects
            self.conv4,
            self.conv5,
            self.conv6,
            self.tran2,
        )