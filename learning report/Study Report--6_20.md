### Study Report --6/20



#### 改进U-Net网络的肺结节分割方法

 钟思华,郭兴明,郑伊能.
计算机工程与应用:1-8[2020-06-20].



由于肺结节等医学图像具有边缘模糊，目标区域小等特点，仅使用原始U-Net对其训练，存在梯度消失、特征利用率低等问题，最终导致模型的分割准确率难以提高。

基于此，该文在U-Net网络结构的基础上，针对不足，提出了一种改进的Dense-Unet网络的肺结节分割算法。



##### 该文章的主要贡献有两个方面：

（1）损失函数：在传统语义分割中所使用的二值交叉熵损失函数的基础上，结合Dice相似系数损失函数，组成混合损失函数。该混合损失函数保证了网络能够稳定且有针对地对难易学习的样本进行优化，从而缓解类不平衡的问题，改善网络分类结果。

（2）网络结构：借鉴了DenseNet中密集连接的概念，在U-Net网络的卷积层之间引入密集连接，将网络中上下卷积层之间的特征结合起来。针对部分小目标区域存在提取特征困难的问题，通过密集连接方式，可以加强网络对特征的传递与利用，同时解决梯度消失的问题。



##### 密集连接模块



![image-20200621101622674](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200621101622674.png)

​                                            密集连接模块  



每个密集连接模块主要包含了两个3×3的卷积层和两次特征融合操作。对于输入密集连接模块的特征图，在每经过一次卷积操作后，所产生的特征图便与最原始的特征图进行融合形成新的特征图，最后再将特征图输入下一个密集连接模块。此外，每个卷积层后面均添加了归一化层和修正线性单元激活层，以此提高网络性能。



##### Dense-Unet网络结构



![image-20200621101819660](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200621101819660.png)

•Dense-Unet由编码器、解码器、分类器和跳跃连接组成。编码器包含了密集连接模块和最大池化层，其中，密集连接模块通过卷积层用于提取图像的语义特征，最大池化层用于特征图的下采样操作，在减少网络运算量的同时增加特征图的感受野，提高图像特征的鲁棒性。对于输入网络的图像，首先会经过密集连接模块进行两次卷积操作，得到尺寸大小为64×64的特征图，随后经由池化操作将特征图的尺寸大小减半。最终，在经过四次卷积和池化操作后，得到4×4大小的特征图。

•解码器部分包含了密集连接模块和反卷积层，其中，反卷积层用于特征图的上采样操作，从而恢复特征图的分辨率。特征图每经过一次反卷积操作，其尺寸大小都增大一倍，最终可以得到大小与输入图像相同的特征图。此外，在编码器和解码器之间通过一个密集连接模块进行连接。

•分类器由1×1卷积层和sigmoid激活层组成，其中，1×1卷积层用于减少特征图的数量，sigmoid激活层用于计算最终的特征图中每个像素的类别，从而输出网络的分割概率图。跳跃连接将网络中的浅层简单特征与深层抽象特征融合起来，从而可以得到更为精细的分割结果。

•

##### 损失函数

•二值交叉熵损失函数：![image-20200621101938601](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200621101938601.png)

g_i 为像素点i的真实类别,p_i 为网络对像素点i的预测结果

二值交叉熵损失函数进行优化，能稳定的将各个类别所对应的梯度进行回传，有效解决网络在反向传播过程中梯度消失的问题。但由于该损失函数在梯度回传过程中对图像上的每一个类别都平等地进行评估，所以对于存在类不平衡问题的图像来说，其中最常见的类别更容易改变网络的优化方向，进而影响最终的分割结果。

•Dice相似系数损失函数：

![image-20200621102009387](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200621102009387.png)

Dice相似系数损失函数能够知道网络通过不断学习，让预测结果逐渐逼近真实结果。但是一旦预测结果中有部分像素预测错误，会导致预测目标的梯度变化剧烈，从而使得网络的训练过程变得困难。

•该文中的混合损失函数：

![image-20200621102104614](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200621102104614.png)