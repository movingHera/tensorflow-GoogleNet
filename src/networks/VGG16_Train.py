from networks.network import Network
from google_net.config import cfg
import tensorflow as tf

class VGG16_Train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.terminals = []
        self.trainable = trainable
        self.n_classes = cfg.TRAIN.N_CLASSES
        self.data = tf.placeholder(tf.float32, shape=(None, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS))
        self.keep_prob = tf.placeholder(tf.float32, shape=())
        self.layers = dict({'data': self.data})
        self.setup()


    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='drop7')
             .fc(self.n_classes, relu=False, name='fc8_new')
             .softmax(name='prob'))
