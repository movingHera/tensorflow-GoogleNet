# 用于分类的深度学习网络

这是一个入门级的分类网络，CNN有GoogLeNet和VGG16两种，其中GooLeNet的训练有单机单卡和单机多卡（默认4卡）两个版本，而VGG16只有单机单卡的版本，但是要改成单机多卡也是和GoogLeNet一样的方法。

目前所使用的数据集是开源的standford car数据集，一共有196个细分类，训练集和测试集大约都是8000张的样子。大概的训练结果如下：

1. GoogLeNet: 测试准确率82%左右
2. VGG16：测试准确率70%左右（这个数据偏低）

## 训练的一些技巧

* 使用pretrained model。首先在网上下载用imagenet训练过的GoogLeNet和VGG16的模型，并且自定义网络的最后一层。对于分类网络而言，最后一层是fc层，有100个类则有100个输出，在我们的问题中，我们需要把该层的输出节点数改为196。在源码中，模型的加载时通过name_scope来进行的，比如第一层卷积层名字为"conv1"，则参数名分别是"conv1/weights"和"conv1/biases"。我们加载pretrained model就是通过这个一一对应关系来加载的，因此我们需要为最后一层重新命名，随便改个"new layer"什么的名字，这样pretrained model中的网络层名称和你定义的名称不一样，它就不会将pretrained model中的最后一层参数给加载进来。