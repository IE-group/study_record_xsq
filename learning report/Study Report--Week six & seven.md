### Study Report--Week six & seven

上周主要忙着做数学建模

我和我的队伍选择了A题，偏向于图像处理，即图像分类+分割，算是参与完成了一个小型的图像处理项目。这里把上周学习到的部分以及这周学习的部分进行总结：

在图像分类方面，我们采取的是基于开源深度学习框架PyTorch编程实现，在resnet18上进行迁移学习。

在这个部分对训练集和测试集做了图像增强：

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomRotation(180, fill=255),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

load数据

```python
images = sorted(glob.glob('/data00/home/wenxin.me/work/vein-seg/croped/*/*/*.jpg'))
total_images = len(images)
train_images = random.sample(images, int(1.0 * total_images))
test_images = [path for path in images if path not in train_images]

train_dataset = VeinDataset(train_images, data_transforms['train'])
test_dataset = VeinDataset(test_images, data_transforms['test'])
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
```

下面是训练模型的部分

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-04
epochs = 64
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        ret, predictions = torch.max(outputs, 1)
        correct_counts = predictions.eq(labels.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i) % 1 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}, Train Acc: {acc:4f}")
    scheduler.step()
```

这里做到了：

- 安排学习率
- 保存最佳模型



再讲图像分割部分

采用了python skimage进行图像处理

用canny算子提取边缘

```python
import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import data,feature

#使用Probabilistic Hough Transform.
image = data.camera()
edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=25)
lines = st.probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)
print(len(lines))
# 创建显示窗口.
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
plt.tight_layout()

#显示原图像
ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()

#显示canny边缘
ax1.imshow(edges, plt.cm.gray)
ax1.set_title('Canny edges')
ax1.set_axis_off()

#用plot绘制出所有的直线
ax2.imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
row2, col2 = image.shape
ax2.axis((0, col2, row2, 0))
ax2.set_title('Probabilistic Hough')
ax2.set_axis_off()
plt.show()
```

删除小区快

```python
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt

#编写一个函数来生成原始二值图像
def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  #生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  #随机数种子
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n)) #高斯滤波
    return mask > mask.mean()

data = microstructure(l=128) #生成测试图片

dst=morphology.remove_small_objects(data,min_size=300,connectivity=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(data, plt.cm.gray, interpolation='nearest')
ax2.imshow(dst,plt.cm.gray,interpolation='nearest')

fig.tight_layout()
plt.show()
```

提取骨架

```python
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt

#编写一个函数，生成测试图像
def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n))
    return mask > mask.mean()

data = microstructure(l=64) #生成测试图像

#计算中轴和距离变换值
skel, distance =morphology.medial_axis(data, return_distance=True)

#中轴上的点到背景像素点的距离
dist_on_skel = distance * skel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
#用光谱色显示中轴
ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
ax2.contour(data, [0.5], colors='w')  #显示轮廓线

fig.tight_layout()
plt.show()
```

