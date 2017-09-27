# 用于分类的深度学习网络

这是一个入门级的分类网络，CNN有GoogLeNet和VGG16两种，其中GooLeNet的训练有单机单卡和单机多卡（默认4卡）两个版本，而VGG16只有单机单卡的版本，但是要改成单机多卡也是和GoogLeNet一样的方法。

目前所使用的数据集是开源的standford car数据集，一共有196个细分类，训练集和测试集大约都是8000张的样子。大概的训练结果如下：

1. GoogLeNet: 测试准确率82%左右
2. VGG16：测试准确率60%左右（这个数据偏低）

如果提示没有"easydict"，使用pip安装即可

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
`dataset.py`

这个函数最重要的两个部分是：a) 加载数据； b) 创建队列管理batch。

* “加载数据”部分：调用read_annotation_file函数获得所有图片的path和labels，然后调用process函数读取图像，对其进行中心裁剪，resize等操作，并且随机改变HSV空间（在训练过程中，测试过程不做数据增强）。
* “创建队列”部分：我们使用FIFOQueue来存储图像路径和标记，另外因为tensorflow对png和jpg图像的读取方式不一样，我们将后缀（mask）也作为队列元素的成员。调用相应的enqueue op并且在feed dict里面装载相应的数据既可以将数据装进队列。
在生成batch的时候，我们使用`tf.train.batch_join`，该函数接受一个tensor list作为输入，根据用户所需要的batch size产生相应大小的数据包。注意由于在队列里存储的是路径，我们需要先将路径转化为图像数据，再装进这个batch_join函数中。

注意事项：
* 在获取数据包之前一定要确保队列中有数据，因此我设置了一个queue len变量来记录当前队列中剩余元素的个数，如果queue len小于batch size，那么就会装填数据。
* tf.train.batch_join支持多线程读取数据，但我亲测发现数据顺序会被打乱，因此目前使用单线程读取数据，简单地来说就是给batch join函数的tensor list是由一个process函数产生的。
* 在将队列中的元素转化为tensor list的时候，尽管我们每次只读取一个元素（单线程），但是要将元素里面的成员（label, mask, path）给提取出来还是需要使用unstack函数，否则进程会卡死，原因不明，照着做就没问题。

`network.py`

这是目前tensorflow中用得很多的一个类，用来创建网络模型，包含了各种网络层的实现（实际上实现是tensorflow做好的，但是它将参数的接口进行了规范化，并且将模型的搭建变得形象）
一般我们创建神经网络，大概都是如下般定义
```python
# Parameters
W1 = tf.Variable(...)
b1 = tf.Variable(...)
W2 = tf.Variable(...)
b2 = tf.Variable(...)
# Connect layers
h1 = conv2d(x, W1, b1)
h2 = conv2d(h1, W2, b2)
```
当网络层数很深（100+）的时候，使用这样的写法写出来的代码可以说是相当难看的。那么network.py里面对这一点做了处理：定义了类的成员变量terminals。terminals记录了当前层的输入，使用feed函数结合相关网络层的名称
来定义terminals的数据。比如当前层名字叫"conv2"，其输入是"conv1"，那么先调用feed("conv1")，再调用conv(...,name="conv2")，那么创建conv2时会自动将terminals作为输入，详情参考装饰器`layer(op)`。

下面简单地介绍一些函数：
* `load`: 加载pretrained model。可以看到是根据op_name + param_name与网络层中的参数对应起来的，ignore_missing我们设置为True，因为我们需要修改最后一层网络，因此最后一层的参数是无法读取的。在加载数据的过程中我们将相关的参数
装载到`pretrained_var_list`中，有两点作用：1. 在调用`tf.global_variables_intializer`的时候，可以声明对这些pretrained的变量不再进行初始化，否则会进行覆盖。 2. 可以方便我们定义对这些参数的梯度处理，比如其学习速率要低于我们自定义的层。
* `get_params`: 获取某一特定网络层的参数，*args一般是"weights"和"biases"。
* `get_output`: 获取某一网络层的输出，在network这个类中，通过layers[name]获得的正是名字为"name"的网络层的输出。
* `conv`: 卷积层，k_h, k_w是窗口大小，s_h和s_w是垂直方向和水平方向的stride大小（窗口滑动幅度，一般都是1），group参数应该是和卷积层拼接有关的，暂时不用理会。
* 像softmax, relu, lrn的使用很简单，这里不说了。
* `concat`: 卷积层拼接，一般选择的是channel拼接，也就是进行拼接的feature map大小是一样的，然后按channel展开连接，所以一般来说axis=3。

`solver.py`

这个函数比较长，是用来管理训练过程的。
* `snapshot`: 保存模型文件，隔若干个iteration保存一次。
* `average_gradients`: 用于单机多卡训练，主要是把各个gpu算得的梯度进行平均，得到每个变量的平均梯度。(输入参数必须是具体的参数值，即调用opt.compute_gradients计算所得梯度，而不能是梯度的op算子（使用tf.gradients可以得到梯度的op算子）)
* `feed_all_gpu`: 为每个gpu生成feed dict，其中"models"存放的是tf.placeholder，在各gpu的网络初始化时就可以看到。
* `train_googlenet_multigpu`: 单机多卡训练GoogLeNet。这里面比较重要的模型的重用，即多卡实际上使用的是同一个模型，这个实现需要依赖于`tf.variable_scope`。这里介绍一下`variable_scope`和`name_scope`的区别:

a) `tf.variable_scope`需结合`tf.get_variable`使用，这时定义的变量只会收到`variable_scope`的影响，而不会受到`name_scope`的影响，也就是变量名只取决于`variable_scope`。另外，`tf.get_variable`定义的变量是不可以有重名的，只能重用。

b) `tf.Variable`，其命名同时受到`tf.variable_scope`和`tf.name_scope`的影响，而且不支持重复使用，命名的时候会自动编号，比如"var:0"和"var:1"，因而支持重名，其实也是不一样的名字。

说回多卡训练。可以看到首先对batch数据进行切分，每个gpu只使用其中一部分数据。然后使用`with tf.device('/gpu:%d' % gpu_id)`来定义各个gpu的网络。实际网络的初始定义只有一次，因此可以看到
`tf.variable_scope`里重用参数当且仅当"gpu_id > 0"。同理，pretrained的模型的加载也只在id为0的gpu上进行。变量models负责记录每块卡的数据和结果，特别是梯度。然后调用`average_gradients`
对梯度进行平均，梯度下降算子`apply_gradient_op`是针对平均梯度的，loss是多卡的平均loss，将`avg_loss_op`作为计算loss的算子。这个模块还包括了测试模型的过程，注意测试的时候会丢掉一些数据（
当不能被一次训练的数据量(payload * ngpus)整除时），因为测试也是使用多卡进行的。

* `train_googlenet_model`: 这是单机单卡训练GoogLeNet的模块。在这里我们可以将最后一层的学习速率设置为其它层的10倍，方法是先将所有变量的梯度op给收集起来，并且将pretrained变量的梯度op分配给
低学习速率的train op，将最后一层变量的梯度收集起来给高学习速率的train op。最后将二者结合在一起就是最终训练的op。测试过程中我们使用了所有的数据，因为是单卡所以这个好处理，对于不能被batch size
整除的部分，则最后一次测试的batch减少即可。另外加了注释的代码里有关于其他输出分支的loss，注意GoogLeNet有三个输出分支，但是在fine tune的时候只需要计算最后那个分支的loss就可以了，所以暂时不适用其他分支的loss。

