# CNN

## 原始版CNN
原始版本的卷积网络结构如下：
```python
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

数据通过两个卷积层+池化层后，进入全连接层，最后输出10个类别的概率。

训练Loss曲线和准确率曲线如下：
<center>
<img src="../res/cnn_base_result.png" width="600">
</center>

最终准确率在62%左右。

## ResNet
复现ResNet18网络，结构如下(由于输出太长，用代码代替)：
```python
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

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 前两层和googlenet一样
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的主体结构：4个模块，每个模块包含两个残差块
        # 第一个模块块不改变通道数和大小
        # 后面的每个模块将上一个模块的通道数翻倍，大小减半
        self.resnet_block1 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.resnet_block2 = nn.Sequential(*resnet_block(64, 128, 2))
        self.resnet_block3 = nn.Sequential(*resnet_block(128, 256, 2))
        self.resnet_block4 = nn.Sequential(*resnet_block(256, 512, 2))
        
        # 全局平均池化层与全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
```

- 网络前两层和GoogleNet类似，使用7x7的卷积核，后接BatchNorm和ReLU激活函数，最好是MaxPool2d池化层。
- 之后是4个ResNet Block，每个Block包含两个Residual Block，每个Residual Block包含两个卷积层和BatchNorm。
- 最后是全局平均池化层和全连接层。