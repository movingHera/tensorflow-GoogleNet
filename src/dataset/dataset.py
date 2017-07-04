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



class DataProducer(object):
    '''
    Loads and processes batches of training images in parallel
    '''
    def __init__(self, train_annotation_filename, test_annotation_filename):
        self.train_image_paths, self.train_labels = self.read_annotation_file(train_annotation_filename, 'train')
        self.test_image_paths, self.test_labels = self.read_annotation_file(test_annotation_filename, 'test')
        # A boolean flag per image indicating whether its a JPEG or PNG 
        self.train_extension_mask = self.create_extension_mask(self.train_image_paths)
        self.test_extension_mask = self.create_extension_mask(self.test_image_paths)        
        self.train_batch_num = cfg.TRAIN.BATCH_NUM
        self.train_batch_size = cfg.TRAIN.BATCH_SIZE
        self.test_batch_num = cfg.TEST.BATCH_NUM
        self.test_batch_size = cfg.TEST.BATCH_SIZE
        
        # self.setup(sess, coord)


    def setup(self, sess, coord):
       


        # Placeholder for image path, label and extension_mask
        self.image_path_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_path')
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='label')
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None,1), name='extension_mask')

        # Create a queue that will contain all image paths
        # Together with labels and extension indicator
        self.train_path_queue = data_flow_ops.FIFOQueue(capacity = 10000, dtypes = [tf.int32, tf.bool, tf.string], shapes=[(1,),(1,),(1,)], name='train_path_queue') 
        self.train_enqueue_path_op = self.train_path_queue.enqueue_many([self.label_placeholder, self.mask_placeholder, self.image_path_placeholder])

        
        self.test_path_queue = data_flow_ops.FIFOQueue(capacity = 10000, dtypes = [tf.int32, tf.bool, tf.string], shapes=[(1,),(1,),(1,)], name='test_path_queue') 
        self.test_enqueue_path_op = self.test_path_queue.enqueue_many([self.label_placeholder, self.mask_placeholder, self.image_path_placeholder])

        
        train_images_and_labels = []
        test_images_and_labels = []

        train_label, train_image = self.process(self.train_path_queue)
        test_label, test_image = self.process(self.test_path_queue)

        train_images_and_labels.append([[train_image], [train_label]])
        test_images_and_labels.append([[test_image], [test_label]])

        image_shape = (cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS)

        self.train_image_batch, self.train_label_batch = tf.train.batch_join(train_images_and_labels, batch_size=self.train_batch_size, shapes=[image_shape, ()], enqueue_many=True, capacity = 4 * self.train_batch_size, allow_smaller_final_batch=True)

        self.test_image_batch, self.test_label_batch = tf.train.batch_join(test_images_and_labels, batch_size=self.test_batch_size, shapes=[image_shape, ()], enqueue_many=True, capacity = 4 * self.test_batch_size, allow_smaller_final_batch=True)
         
        
        tf.train.start_queue_runners(coord=coord, sess=sess)

        #return train_image_batch, train_label_batch, test_image_batch, test_label_batch

        #self.queue_runner = tf.train.queue_runner.QueueRunner(self.path_queue, [self.enqueue_path_op]*4)
  
    def get_batch_data(self, sess, mask):
        '''
        extract batch of images
        mask: 'train' or 'test'
        '''
        image_batch_array = None
        label_batch_array = None
        
        if mask == 'train':
            image_batch_array, label_batch_array = sess.run([self.train_image_batch, self.train_label_batch])
        else:

            image_batch_array, label_batch_array = sess.run([self.test_image_batch, self.test_label_batch])
        
        #image_batch = self.train_image_batch if mask == 'train' else self.test_image_batch
        #label_batch = self.test_label_batch if mask == 'train' else self.test_label_batch


        image_batch_array = np.asarray(image_batch_array)
        label_batch_array = np.asarray(label_batch_array)

        return image_batch_array, label_batch_array




    def enqueue_image_paths(self, sess, mask):
        '''
        mask: 'train' or 'test'
        '''
        image_paths = self.train_image_paths if mask == 'train' else self.test_image_paths
        labels = self.train_labels if mask == 'train' else self.test_labels
        extension_mask = self.train_extension_mask if mask == 'train' else self.test_extension_mask

        # the total number of files 
        total_file_num = len(image_paths)

        # the number of files in each epoch
        file_num_per_epoch = self.train_batch_size * self.train_batch_num if mask == 'train' else self.test_batch_size * self.test_batch_num

        # the enqueue_op
        enqueue_path_op = self.train_enqueue_path_op if mask == 'train' else self.test_enqueue_path_op

        # generate shuffle indexes for files in this epoch
        shuffle_index = np.random.randint(total_file_num, size=file_num_per_epoch)

        image_path_array = np.array(image_paths)[shuffle_index]
        label_array = np.array(labels)[shuffle_index]
        mask_array = np.array(extension_mask)[shuffle_index]

        image_path_array = np.reshape(np.expand_dims(image_path_array, 1), (-1,1))
        label_array = np.reshape(np.expand_dims(label_array, 1), (-1,1))
        mask_array = np.reshape(np.expand_dims(mask_array, 1), (-1,1))

        # print "========"
        # print label_array.shape
        # print "========"

        # Queue all paths
        sess.run(enqueue_path_op, {self.label_placeholder: label_array, self.mask_placeholder: mask_array, self.image_path_placeholder: image_path_array})
        

        # Close the path queue
        # session.run(self.close_path_queue_op)


    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(is_jpeg, lambda: tf.image.decode_jpeg(file_data, channels=cfg.PREPROCESS.CHANNELS), lambda: tf.image.decode_png(file_data, channels=cfg.PREPROCESS.CHANNELS))
        if cfg.PREPROCESS.EXPECTS_BGR:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels
            # img = tf.reverse(img, [False, False, True])
            # tensorflow 1.0 tf.reverse api
            img = tf.reverse(img, [2]) 
        return img

    def process(self, path_queue):
        # Dequeue a single image path
        label, is_jpeg, image_path = path_queue.dequeue()
        image_path = tf.unstack(image_path)[0]
        #print image_path[0].shape
        is_jpeg = tf.unstack(is_jpeg)[0]
        label = tf.unstack(label)[0]
        #for image_path in tf.unstack(image_path):
        #    print "----------"
        #print image_path
        #image_path = sess.run(image_path)
        #print image_path[0]
        #image_path = tf.constant(image_path)
        #for ff in tf.unstack(image_path):
        #    image_path = ff
        #    print sess.run(ff)
        #print image_path
        #is_jpeg = sess.run(is_jpeg)
        #is_jpeg = tf.constant(is_jpeg)
        #for ll in tf.unstack(is_jpeg):
        #    is_jpeg = ll
        #label = sess.run(label)
        #label = tf.constant(label)
        #label = tf.unstack(label)

        img = self.load_image(image_path, is_jpeg)
        processed_img = process_image(img=img, 
                                    scale=cfg.PREPROCESS.SCALE_SIZE,
                                    isotropic=cfg.PREPROCESS.ISOTROPIC,
                                    crop=cfg.PREPROCESS.CROP_SIZE,
                                    mean=cfg.PREPROCESS.MEAN)
        # Return the processed image, along with its label
        #print processed_img
        return label, processed_img

    def read_annotation_file(self, annotation_filename, mark):
        # Here we should add the root path the image name
        # We only cares about the labels and image paths now
        # mark: 'train' or 'test'

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
                        image_path = os.path.join(cfg.DATA_DIR, mark, image_path)
                        assert os.path.exists(image_path),\
                                'Path does not exist: {}'.format(image_path)
                        image_paths.append(image_path)
                line = file_to_read.readline()
        return image_paths, labels


    @staticmethod
    def create_extension_mask(paths):
        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False
        return [is_jpeg(p) for p in paths]

