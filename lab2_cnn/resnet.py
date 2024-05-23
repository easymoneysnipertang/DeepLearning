import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:  
            # 用于修改图片大小(stride>1)和通道数
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 图片大小减半，通道数翻倍
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 保持图片大小和通道数不变
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # 前两层和googlenet一样
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的主体结构：4个模块，每个模块包含多个残差块
        # 第一个模块块不改变通道数和大小
        # 后面的每个模块将上一个模块的通道数翻倍，大小减半
        self.resnet_block1 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.resnet_block2 = nn.Sequential(*resnet_block(64, 128, 2))
        self.resnet_block3 = nn.Sequential(*resnet_block(128, 256, 2))
        self.resnet_block4 = nn.Sequential(*resnet_block(256, 512, 2))
        
        # 全局平均池化层与全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.outputs = {}

    def forward(self, X):
        X = self.conv(X)
        self.outputs["After initial conv"] = X.shape
        X = self.bn(X)
        self.outputs["After batch norm"] = X.shape
        X = self.relu(X)
        X = self.mp(X)
        self.outputs["After max pool"] = X.shape

        X = self.resnet_block1(X)
        self.outputs["After resnet block 1"] = X.shape
        X = self.resnet_block2(X)
        self.outputs["After resnet block 2"] = X.shape
        X = self.resnet_block3(X)
        self.outputs["After resnet block 3"] = X.shape
        X = self.resnet_block4(X)
        self.outputs["After resnet block 4"] = X.shape

        X = self.global_avg_pool(X)
        self.outputs["After global avg pool"] = X.shape
        X = X.reshape(X.shape[0], -1)
        self.outputs["After reshape"] = X.shape
        X = self.fc(X)
        self.outputs["After fc"] = X.shape
        return X

    def get_layer_shapes(self):
        return self.outputs
    

if __name__ == "__main__":
    net = ResNet18(3, 10)
    # X = torch.randn(size=(1, 3, 224, 224))
    # net(X)
    # output_shapes = net.get_layer_shapes()
    # for key, value in output_shapes.items():
    #     print(f"{key}: {value}")

    summary(net, input_size=(3, 224, 224), device='cpu')
    