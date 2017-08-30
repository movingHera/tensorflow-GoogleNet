'''
Class for handling the vehicle dataset
'''

import os
import os.path as osp
import numpy as np
import tensorflow as tf
from google_net.config import cfg
from tensorflow.python.ops import data_flow_ops

def process_image(img, scale, isotropic, crop, mean):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken. 
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])
    img = tf.image.resize_images(img, (new_shape[0], new_shape[1]))
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.stack([offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean


def distort_color(image, scope=None):
    '''Distort the color of the image

    Args:
        image: Tensor containing single image.
        scope: Optional scope for op_scope
    Returns:
        color-distorted image
    '''
    with tf.op_scope([image], scope, 'distort_color'):
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        
        return image


class DataProducer(object):
    '''
    Loads and processes batches of training images in parallel
    '''
    def __init__(self, dataset_name, image_set, batch_size=None):
        '''Initialize data producer

        Args:
            dataset_name: the name of the training/test dataset
            image_set: 'train' or 'test'
            batch_size: the size of data fetched each time, if none, the batch
                        size is the number of data by default 
        '''
        self.__image_set = image_set
        self.__annotation_filename = dataset_name + '_' + image_set + \
                                    '_annotation.txt'
        self.__image_paths, self.__image_labels = \
                                    self.read_annotation_file(image_set)
        
        # A boolean flag per image indicating whether its a JPEG or PNG 
        self.__extension_mask = self.create_extension_mask(self.__image_paths)
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.__image_paths)

                
        # self.setup(sess, coord)

    def get_data_num(self):
        return len(self.__image_paths)

    def setup(self, sess, coord):
        # Placeholder for image path, label and extension_mask
        self.image_path_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_path')
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='label')
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None,1), name='extension_mask')

        # Create a queue that will contain all image paths
        # Together with labels and extension indicator
        self.__path_queue = data_flow_ops.FIFOQueue(capacity = 10000, dtypes = [tf.int32, tf.bool, tf.string], shapes=[(1,),(1,),(1,)], name='path_queue') 
        self.__enqueue_path_op = self.train_path_queue.enqueue_many([self.label_placeholder, self.mask_placeholder, self.image_path_placeholder])

        images_and_labels = []
        
        label, image = self.process(self.__path_queue)

        images_and_labels.append([[image], [label]])

        image_shape = (cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS)

        self.__image_batch, self.__label_batch = tf.train.batch_join(images_and_labels, batch_size=self.batch_size, shapes=[image_shape, ()], enqueue_many=True, capacity = 4 * self.batch_size, allow_smaller_final_batch=True)
        
        # The state of the queue
        self.__queue_len = 0

        tf.train.start_queue_runners(coord=coord, sess=sess)

        #return train_image_batch, train_label_batch, test_image_batch, test_label_batch
        #self.queue_runner = tf.train.queue_runner.QueueRunner(self.path_queue, [self.enqueue_path_op]*4)
  
    def get_batch_data(self, sess):
        ''' extract batch of images (size of self.batch_size)
        
        Returns:
            image_batch_array: (b, h, w, c) format ndarray
            label_batch_array: (b, 1) format ndarray
        '''

        if self.__queue_len < self.batch_size:
            self.enqueue_image_paths(sess)

        image_batch_array, label_batch_array = sess.run([self.__image_batch,
            self.__label_batch])
        
        self.__queue_len -= self.batch_size

        image_batch_array = np.asarray(image_batch_array)
        label_batch_array = np.asarray(label_batch_array)

        return image_batch_array, label_batch_array


    def enqueue_image_paths(self, sess):
        '''Enqueue image paths

        It's needed when the elements in the queue are not enough
        
        If we want to get elements from "hungry" queue, the session will be
        blocked. Do remember to make sure there are redandunt elements in the
        queue before fetching data.
        '''

        image_paths = self.__image_paths
        labels = self.__labels 
        extension_mask = self.__extension_mask 

        # the total number of files 
        total_file_num = len(image_paths)

        # the enqueue_op
        enqueue_path_op = self.__enqueue_path_op

        # generate shuffle indexes for files in this epoch
        shuffle_index = np.random.randint(total_file_num, size=total_file_num)

        image_path_array = np.array(image_paths)[shuffle_index]
        label_array = np.array(labels)[shuffle_index]
        mask_array = np.array(extension_mask)[shuffle_index]

        # expand vector to matrix
        image_path_array = np.expand_dims(image_path_array, 1)
        label_array = np.expand_dims(label_array, 1)
        mask_array = np.expand_dims(mask_array, 1)

        # Queue all paths
        sess.run(enqueue_path_op, {self.label_placeholder: label_array, self.mask_placeholder: mask_array, self.image_path_placeholder: image_path_array})
        
        self.__queue_len += total_file_num
        # Close the path queue
        # session.run(self.close_path_queue_op)


    def load_image(self, image_path, is_jpeg):
        ''' Read the image from image path

        If the image is for training, we apply distortion
        
        Args:
            image_path: absolute path of image
            is_jpeg: denote the format of image
        Return:
            img: image array with format RGB or BGR
        '''
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(is_jpeg, lambda: tf.image.decode_jpeg(file_data, channels=cfg.PREPROCESS.CHANNELS), lambda: tf.image.decode_png(file_data, channels=cfg.PREPROCESS.CHANNELS))
        if self.__image_set == 'train':
            # Distort the image in training process
            img = distort_color(img)
        
        if cfg.PREPROCESS.EXPECTS_BGR:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels
            # img = tf.reverse(img, [False, False, True])
            # tensorflow 1.0 tf.reverse api
            img = tf.reverse(img, [2]) 
        return img

    def process(self, path_queue):
        '''Dequeue a single image path
        
        Args:
            path_queue: a queue containing tensor list [label, is_jpeg,
                        image_path]
        Returns:
            label, processed_image (resized and distorted is needed)
        '''
        label, is_jpeg, image_path = path_queue.dequeue()
        # Bug, there is exactly one label, one image_path,
        # however, to get the element, we should use tf.unstack...
        image_path = tf.unstack(image_path)[0]
        #print image_path[0].shape
        is_jpeg = tf.unstack(is_jpeg)[0]
        label = tf.unstack(label)[0]
       
        img = self.load_image(image_path, is_jpeg)
        processed_img = process_image(img=img, 
                                    scale=cfg.PREPROCESS.SCALE_SIZE,
                                    isotropic=cfg.PREPROCESS.ISOTROPIC,
                                    crop=cfg.PREPROCESS.CROP_SIZE,
                                    mean=cfg.PREPROCESS.MEAN)
        # Return the processed image, along with its label
        #print processed_img
        return label, processed_img

    def read_annotation_file(self, annotation_filename):
        '''Read annotation file from annotation_filename

        It's only for the standford cars dataset
        The first 4 elements are gt boxes info
        The 5th element is the classification info
        The 6th element is the image name

        Args:
            annotation_filename: absolute path
        Returns:
            image_paths: absolute image paths
            labels: classfication label
        '''

        image_paths = []
        labels = []
        assert os.path.exists(annotation_filename),\
                'Path does not exist: {}'.format(annotation_filename)
        with open(annotation_filename, 'r') as file_to_read:
            line = file_to_read.readline()
            while line:
                data = line.split()
                for (index, value) in enumerate(data):
                    image_path = None
                    label = None
                    if index == 4:
                        label = int(value)-1
                        labels.append(label)
                    if index == 5:
                        value = value.strip('\n')  # eliminate the newline character '\n'
                        image_path = str(value)
                        image_path = os.path.join(cfg.DATA_DIR,
                                'images', self.__image_set, image_path)
                        assert os.path.exists(image_path),\
                                'Path does not exist: {}'.format(image_path)
                        image_paths.append(image_path)
                line = file_to_read.readline()

        return image_paths, labels


    @staticmethod
    def create_extension_mask(paths):
        '''Judge whether the image ext is jpg or png
    
        Args:
            paths: list of image paths
        Returns:
            list of bool variables, denoting the ext of each image in paths
        '''
        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False
        return [is_jpeg(p) for p in paths]

