import torch.nn as nn
from torchsummary import summary

def conv_bn(in_channels, num_channels, stride):
    # conv2d + batchnorm2d + relu
    return nn.Sequential(
        nn.Conv2d(in_channels, num_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(inplace=True)
        )


def conv_dw(in_channels, num_channels, stride):
    return nn.Sequential(
        # depthwise，对每个输入通道分别进行卷积
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        # pointwise，实现通道之间的信息交互
        nn.Conv2d(in_channels, num_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(inplace=True),
        )


class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(
            conv_bn(in_channels, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__=='__main__':
    model = MobileNetV1(in_channels=3, num_classes=10)
    summary(model, input_size=(3, 224, 224), device='cpu')