import _init_paths
from solver.solver import SolverWrapper
from networks.GoogLeNet_Train import GoogLeNet_Train
import os
import tensorflow as tf
from config.config import cfg

TRAINED_MODEL = 'googlenet'
# TRAINED_MODEL = 'vgg16'

def train():
    '''We use googlenet model/vgg model to do the classification task
    '''
    
    # The output directory for these checkpoint files
    output_dir = cfg.TRAIN.OUTPUT_DIR

    
    if TRAINED_MODEL == 'googlenet':
        pretrained_googlenet_filename = os.path.join(
                cfg.TRAIN.PRETRAINED_MODEL_DIR, 'googlenet.npy')
        assert os.path.exists(pretrained_googlenet_filename), \
            'Path does not exist: {}'.format(pretrained_googlenet_filename)
        # solver handle
        solver = SolverWrapper(output_dir, train_annotation_filename, 
                test_annotation_filename, pretrained_googlenet_filename)
        
        #solver.train_googlenet_model()
        solver.train_googlenet_multigpu()

    else:
        pretrained_vgg16_filename = os.path.join(
                cfg.TRAIN.PRETRAINED_MODEL_DIR, 'vgg16.npy')
        assert os.path.exists(pretrained_vgg16_filename), \
            'Path does not exist: {}'.format(pretrained_vgg16_filename)
        # solver handle
        solver = SolverWrapper(output_dir, train_annotation_filename, 
                test_annotation_filename, pretrained_vgg16_filename)
        solver.train_vgg16_model()


if __name__ == '__main__':
    train()
