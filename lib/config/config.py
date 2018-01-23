import os
import os.path as osp
import numpy as np
from distutils import spawn
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# preprocess for images
__C.PREPROCESS = edict()

# The image should be scaled to this size first during preprocessing
__C.PREPROCESS.SCALE_SIZE = 256

# Whether the model expects the rescaling to be isotropic
__C.PREPROCESS.ISOTROPIC = False

# A square crop of this dimension is expected by this model
__C.PREPROCESS.CROP_SIZE = 224

# The number of channels in the input image expected by this model
__C.PREPROCESS.CHANNELS = 3

# The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
# The values below are ordered BGR, as many Caffe models are trained in this order.
__C.PREPROCESS.MEAN = np.array([104., 117., 124.])

# Whether the model expects images to be in BGR order
__C.PREPROCESS.EXPECTS_BGR = True


# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data', 'images'))

__C.ANNOTATION_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data', 'annotations'))

__C.COARSE_CLASSES = ['SUV', 'Sedan', 'Coupe', 'Convertible', 'Pickup', 'Hatchback', 'Wagon', 'Van']

__C.TRAIN = edict()

# The recommended batch size for this model
__C.TRAIN.BATCH_SIZE = 256

# The number of classes in car dataset
__C.TRAIN.N_CLASSES = 196

# The infix of snap shot file
__C.TRAIN.SNAPSHOT_INFIX = ''

# The prefix of snap shot file
__C.TRAIN.SNAPSHOT_PREFIX = 'GoogLeNet'

# Learning rate
__C.TRAIN.BASE_LEARNING_RATE = 0.001

# Step size
__C.TRAIN.STEP_SIZE = 25000

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# The number of epochs
__C.TRAIN.NUM_EPOCHS = 200

# The interval epochs to catch the snapshot
# We plan to store 5 snapshots
__C.TRAIN.SNAPSHOT_EPOCHS = __C.TRAIN.NUM_EPOCHS/5

# The directory of pretrained GoogLeNet model
__C.TRAIN.PRETRAINED_MODEL_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'pretrained_model'))

# The output directory for snapshot files
__C.TRAIN.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output', 'train'))



__C.TEST = edict()

# The frequency to test the model
__C.TEST.EPOCHS = 1

# Test batch size
__C.TEST.BATCH_SIZE = 32
