### study report--week five

##### 在卷积神经网络上添加分类器

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

##### 在MNIST 图像上训练卷积神经网络

```python
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
```

##### 0填充（Padding）

如何使得输出尺寸与输入保持一致呢？扩充边界，边界都填0值像素。

##### 降采样

池化是降采样的一种手段，其实步长大于1的卷积也能达到降采样的效果。
 (1) 均值池化：对池化区域内的像素点取均值，这种方法得到的特征数据对背景信息更敏感。
 (2) 最大池化：对池化区域内所有像素点取最大值，这种方法得到的特征对纹理特征信息更加敏感。

##### tensorflow相关函数

tf.nn.conv2d
tf.nn.max_pool
tf.nn.dropout
With probability rate, drops elements of x. Input that are kept are scaled up by 1 / (1 - rate), otherwise outputs 0. The scaling is so that the expected sum is unchanged.（丢掉x中rate比例的的元素。没掉丢的输入扩大为1/(1-rate)倍，丢掉的元素值变为0。这个缩放，期望和不变。）

##### 学习链接：

https://blog.csdn.net/weixin_43423455/article/details/98585206

https://blog.csdn.net/weixin_43067754/article/details/89577445

https://blog.csdn.net/yukinoai/article/details/84421560