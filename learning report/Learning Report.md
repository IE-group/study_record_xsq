### Learning Report

VGGNet的具体例子

VGGNet的主要特点在于：**(1)网络很深；(2)卷积层中使用的卷积核很小，且都是3\*3的卷积核。**

### 代码模块

#### **VGG 的 PyTorch 实现**

##### 1.CNN部分

```python

# 卷积层构造函数
# 传入的 cfg 参数表示层类型和层结构
# batch_norm 表示是否使用 BN

def make_layers(cfg, batch_norm=False):
    layers = []
    # 默认输入维度为 3
    in_channels = 3
    for v in cfg:
        # M 表示 MaxPooling 层, size=2, stride=2
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # v 表示这一卷积层的输出维度
            # in_channels 的初始值为 3，之后随着卷积层的增加而变化
            # 全用 3*3 卷积，步长为 1，padding 为 1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 如果选择 BN 模式，就在卷积层和激活函数之间加入 BN 层
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # 更新卷积层的输入维度
            in_channels = v
    # 返回一个 nn 层序列
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```

##### 2.VGG 网络

```python

# VGG 类
# 输入参数 features 是 CNN 部分的网络结构
# num_classes 表示分类数量
# init_weights 表示初始化模式
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights =True):
        super(VGG, self).__init__() # 必加
        # CNN 部分
        self.features = features
        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    # 前向传播
    def forward(self, x):
        # CNN
        x = self.features(x)
        # 展平输入维度
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.classifier(x)
        return x
    # 初始化参数函数
    # 是 PyTorch 中最常用的初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            # 初始化卷积层参数
            # 使用针对 ReLU 的初始化方法：均值为 0，方差为 2/n
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # BN 层，全 1 初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # 全连接层，均值为 0，方差为 0.01
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
```

##### 3.生成 VGG 网络

```python

# 普通 VGG-16
def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 如果使用预训练模型，就把参数初始化设为 False
    if pretrained:
        kwargs['init_weights'] = False
    # VGG-16 对应结构 D
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        # 预训练模型参数加载

        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
 
# 使用 BN 的 VGG-16
def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model
```

