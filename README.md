# 用于分类的深度学习网络

这是一个入门级的分类网络，CNN有GoogLeNet和VGG16两种，其中GooLeNet的训练有单机单卡和单机多卡（默认4卡）两个版本，而VGG16只有单机单卡的版本，但是要改成单机多卡也是和GoogLeNet一样的方法。

目前所使用的数据集是开源的standford car数据集，一共有196个细分类，训练集和测试集大约都是8000张的样子。大概的训练结果如下：

1. GoogLeNet: 测试准确率82%左右
2. VGG16：测试准确率60%左右（这个数据偏低）

## 训练的一些技巧

* 使用pretrained model。首先在网上下载用imagenet训练过的GoogLeNet和VGG16的模型，并且自定义网络的最后一层。对于分类网络而言，最后一层是fc层，有100个类则有100个输出，在我们的问题中，我们需要把该层的输出节点数改为196。在源码中，模型的加载时通过name_scope来进行的，比如在pretrained model中第一层卷积层名字为"conv1"，则tensorflow网络中的第一层卷积层名字也必须是"conv1"才能加载模型数据。我们只希望加载最后一层之前的网络参数，因此对于最后一层，我们要重新命名，随便改个"new layer"什么的名字就可以了。
* 使用数据增强。由于输入到网络中图片的大小都是统一规格的224x224，对于大的图片，我们选取中间部分。因此我们所做的图像增强一般就是改变图像的HSV颜色空间，tensorflow为我们提供了相应的函数。
* 使用不同的学习速率。除了最后一层，其余层使用的都是pretrained的参数，因此我们尽可能地多改变我们自定义的网络层，而减少对其它层的改变。因此我们需要将最后一层的学习速率提高到其它层的10倍，我们在单机单卡的训练中体现了这一点思想。对于单机多卡，尚有Bug未解决，暂时是所有层统一学习速率。

## 代码讲解

### 代码结构
1. `lib/`：有关数据产生、网络定义以及训练管理器的代码
    * `dataset/`: 数据产生模块，包括读取annotation file，进行数据增强，将image和label打包成batch供训练使用。
    * `config/`: 配置文件，相当于全局变量，超参数以及路径等定义就在这里。
    * `networks/`: 包括一个基类network(实现卷积、全连接等层的功能)，还有GoogLeNet和VGG16的网络结构定义（继承network）。
    * `solver/`: 训练管理器，包括读取数据、创建网络、创建优化器、进行模型存储等。
    * `utils/`: 提供了计时器timer
    
2. `tools/`：启动网络的训练。要进行网络训练直接运行`python train.py`就可以了。

### 代码说明
1. dataset.py

这个函数最重要的两个部分是：a) 加载数据； b) 创建队列管理batch。

* “加载数据”部分：调用read_annotation_file函数获得所有图片的path和labels，然后调用process函数读取图像，对其进行中心裁剪，resize等操作，并且随机改变HSV空间（在训练过程中，测试过程不做数据增强）。
* “创建队列”部分：我们使用FIFOQueue来存储图像路径和标记，另外因为tensorflow对png和jpg图像的读取方式不一样，我们将后缀（mask）也作为队列元素的成员。调用相应的enqueue op并且在feed dict里面装载相应的数据既可以将数据装进队列。
在生成batch的时候，我们使用`tf.train.batch_join`，该函数接受一个tensor list作为输入，根据用户所需要的batch size产生相应大小的数据包。注意由于在队列里存储的是路径，我们需要先将路径转化为图像数据，再装进这个batch_join函数中。

注意事项：
* 在获取数据包之前一定要确保队列中有数据，因此我设置了一个queue len变量来记录当前队列中剩余元素的个数，如果queue len小于batch size，那么就会装填数据。
* tf.train.batch_join支持多线程读取数据，但我亲测发现数据顺序会被打乱，因此目前使用单线程读取数据，简单地来说就是给batch join函数的tensor list是由一个process函数产生的。
* 在将队列中的元素转化为tensor list的时候，尽管我们每次只读取一个元素（单线程），但是要将元素里面的成员（label, mask, path）给提取出来还是需要使用unstack函数，否则进程会卡死，原因不明，照着做就没问题。

2. network.py

这是目前tensorflow中用得很多的一个类，用来创建网络模型，包含了各种网络层的实现（实际上实现是tensorflow做好的，但是它将参数的接口进行了规范化，并且将模型的搭建变得形象）
一般我们创建神经网络，大概都是如下般定义
```python
a = c
```
