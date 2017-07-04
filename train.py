import _init_paths
from networks.fine_tune import SolverWrapper
from networks.GoogLeNet_Train import GoogLeNet_Train
import os
import tensorflow as tf
from google_net.config import cfg

def train():
    # We fine tune the GoogLeNet model
    # The three fc layers have been modified
    #network = GoogLeNet_Train()

    # The pretrained GoogLeNet model by imagenet dataset
    # However, we dispense the fc layers
    pretrained_googlenet_filename = os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, 'googlenet.npy')
    assert os.path.exists(pretrained_googlenet_filename), \
            'Path does not exist: {}'.format(pretrained_googlenet_filename)

    pretrained_vgg16_filename = os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, 'vgg16.npy')
    assert os.path.exists(pretrained_vgg16_filename), \
            'Path does not exist: {}'.format(pretrained_vgg16_filename)
    # The saver to save checkpoints files (snapshot)
    #saver = tf.train.Saver(max_to_keep=100)

    # The output directory for these checkpoint files
    output_dir = cfg.TRAIN.OUTPUT_DIR

    # Annotation filename
    train_annotation_filename = 'car_train_annotation.txt'
    test_annotation_filename = 'car_test_annotation.txt'

    training_model = 'googlenet'

    if training_model == 'googlenet':
        # solver handle
        solver = SolverWrapper(output_dir, train_annotation_filename, test_annotation_filename, pretrained_googlenet_filename)
        solver.train_googlenet_model()
    else:
        # solver handle
        solver = SolverWrapper(output_dir, train_annotation_filename, test_annotation_filename, pretrained_vgg16_filename)
        solver.train_vgg16_model()

    #solver.queue_test()

if __name__ == '__main__':
    train()
