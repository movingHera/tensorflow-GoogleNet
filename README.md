# 用于分类的深度学习网络

这是一个入门级的分类网络，CNN有GoogLeNet和VGG16两种，其中GooLeNet的训练有单机单卡和单机多卡（默认4卡）两个版本，而VGG16只有单机单卡的版本，但是要改成单机多卡也是和GoogLeNet一样的方法。

目前所使用的数据集是开源的standford car数据集，一共有196个细分类，训练集和测试集大约都是8000张的样子。大概的训练结果如下：

1. GoogLeNet: 测试准确率82%左右
2. VGG16：测试准确率70%左右（这个数据偏低）

## 训练的一些技巧

* 使用pretrained model。首先在网上下载用imagenet训练过的GoogLeNet和VGG16的模型，并且自定义网络的最后一层。对于分类网络而言，最后一层是fc层，有100个类则有100个输出，在我们的问题中，我们需要把该层的输出节点数改为196。在源码中，模型的加载时通过name_scope来进行的，比如在pretrained model中第一层卷积层名字为"conv1"，则tensorflow网络中的第一层卷积层名字也必须是"conv1"才能加载模型数据。我们只希望加载最后一层之前的网络参数，因此对于最后一层，我们要重新命名，随便改个"new layer"什么的名字就可以了。
* 使用数据增强。由于输入到网络中图片的大小都是统一规格的224x224，对于大的图片，我们选取中间部分。因此我们所做的图像增强一般就是改变图像的HSV颜色空间，tensorflow为我们提供了相应的函数。
* 使用不同的学习速率。除了最后一层，其余层使用的都是pretrained的参数，因此我们尽可能地多改变我们自定义的网络层，而减少对其它层的改变。因此我们需要将最后一层的学习速率提高到其它层的10倍，我们在单机单卡的训练中体现了这一点思想。对于单机多卡，尚有Bug未解决，暂时是所有层统一学习速率。