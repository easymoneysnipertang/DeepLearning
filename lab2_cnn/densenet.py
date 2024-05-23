import torch
from torch import nn
from torchsummary import summary


def conv_block(input_channels, num_channels):
    # 批量归一化层(BatchNorm2d)+激活函数(ReLU)+卷积层(Conv2d)
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    # 一个稠密块由多个conv_block组成，每块使用相同的输出通道数
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
    

def transition_block(input_channels, num_channels):
    # 过渡层用来控制模型的复杂度
    # 通过1x1卷积层来减小通道数，使用步幅为2的平均池化层减半高和宽
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, in_channels, num_channels, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        # DenseNet首先使用同ResNet一样的单卷积层和最大池化层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2,
                      padding=3),
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # DenseNet使用4个稠密块，每个使用4个卷积层，与ResNet的主体结构类似
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        # 每个稠密块通道数增加growth_rate*4
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间加入一个过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        
        # DenseNet最后接上全局池化层和全连接层来输出
        self.b2 = nn.Sequential(*blks, nn.BatchNorm2d(num_channels), nn.ReLU(),
                                 nn.AdaptiveMaxPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(num_channels, num_classes))

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        return X
    

if __name__ == '__main__':
    net = DenseNet(3, 64, 32, 10)
    # X = torch.randn(1, 3, 224, 224)
    # out = net(X)
    # print(out.shape)  # torch.Size([4, 10])
    summary(net, input_size=(3, 224, 224), device='cpu')
        