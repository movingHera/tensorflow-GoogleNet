from google_net.config import cfg
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import time
from dataset.dataset import DataProducer
from networks.GoogLeNet_Train import GoogLeNet_Train
from networks.VGG16_Train import VGG16_Train

class SolverWrapper(object):
    '''
    A solver to control the training process and execute snapshotting process
    '''

    def __init__(self, output_dir, train_annotation_filename, test_annotation_filename, pretrained_model=None):
        #self.net = network
        self.pretrained_model = pretrained_model
        #self.saver = saver
        self.output_dir = output_dir
        self.train_annotation_filename = os.path.join(cfg.ANNOTATION_DIR, train_annotation_filename)
        self.test_annotation_filename = os.path.join(cfg.ANNOTATION_DIR, test_annotation_filename)

    def snapshot(self, sess, iter):
        '''
        Take a snapshot of the network
        '''
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # the saved filename
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename, global_step=iter+1)
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_googlenet_model(self):
        # we only allow 80% usage of gpu
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        coord = tf.train.Coordinator()
        
        
        with sess.as_default():
            # Three solftmax layers
            
            # start the data producer
            data_producer = DataProducer(self.train_annotation_filename, self.test_annotation_filename)
            
            data_producer.setup(sess, coord)
            
            image_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS))

            label_placeholder = tf.placeholder(tf.int64, shape=(None, ))

            self.net = GoogLeNet_Train()
            
            #loss1_logits = self.net.get_output('loss1_classifier')
            #loss2_logits = self.net.get_output('loss2_classifier')
            loss3_logits = self.net.get_output('loss3_classifier_new')


            # define the loss function
            #cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #                labels=label_placeholder, logits=loss1_logits, name='cross_entropy1_per_example')
            #cross_entropy_mean1 = tf.reduce_mean(cross_entropy1, name='cross_entropy1')

            #cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #                labels=label_placeholder, logits=loss2_logits, name='cross_entropy2_per_example')
            #cross_entropy_mean2 = tf.reduce_mean(cross_entropy2, name='cross_entropy2')

            cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=label_placeholder, logits=loss3_logits, name='cross_entropy3_per_example')
            
            cross_entropy_mean3 = tf.reduce_mean(cross_entropy3, name='cross_entropy3')
                    
            #total_loss = 0.3 * cross_entropy_mean1 + 0.3 * cross_entropy_mean2 + 0.4 * cross_entropy_mean3
  
            total_loss = cross_entropy_mean3


            # prediction accuracy
            pred_prob = self.net.get_output('prob3')
            correct_prediction = tf.equal(tf.argmax(pred_prob, 1), label_placeholder)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# initialize variables
          
            # initialize some variable
            self.saver = tf.train.Saver(max_to_keep=100)

           
            # set 10x learning rate for the last fc layer
            pretrained_var = self.net.get_pretrained_variable_list(self.pretrained_model)
            last_var = self.net.get_variable('loss3_classifier_new')
            
            global_step = tf.Variable(0, trainable=False)
            
            momemtum = cfg.TRAIN.MOMEMTUM

            lr_small = tf.train.exponential_decay(cfg.TRAIN.BASE_LEARNING_RATE, global_step, cfg.TRAIN.STEP_SIZE, 0.1, staircase=True)

            lr_large = tf.train.exponential_decay(10*cfg.TRAIN.BASE_LEARNING_RATE, global_step, cfg.TRAIN.STEP_SIZE, 0.1, staircase=True)
 
            opt1 = tf.train.MomentumOptimizer(lr_small, momemtum) 
            opt2 = tf.train.MomentumOptimizer(lr_large, momemtum)

            grads = tf.gradients(total_loss, pretrained_var + last_var)

            grads1 = grads[:len(pretrained_var)]
            grads2 = grads[len(pretrained_var):]

            train_op1 = opt1.apply_gradients(zip(grads1, pretrained_var), global_step=global_step)
            train_op2 = opt2.apply_gradients(zip(grads2, last_var), global_step=global_step)

            train_op = tf.group(train_op1, train_op2)

            #train_op = tf.train.MomentumOptimizer(lr, momemtum).minimize(total_loss, global_step=global_step)
            
            timer = Timer()
            
            sess.run(tf.global_variables_initializer())
            
            # load pretrained variables 
            if self.pretrained_model is not None:
                print ('Loading pretrained model '
                        'weights from {:s}').format(self.pretrained_model)
                # note that we modify the structure of googlenet
                # concretely, the terminal classifed classes are reset
                self.net.load(self.pretrained_model, sess)



            print('Training Executing!')
            
            
            iter = 0
            test_epoch = 0
            print('Go to epoch!')
            for epoch in range(cfg.TRAIN.NUM_EPOCHS):
                '''
                At the begining of each epoch, all image paths should be enqueued
                Then we run serveral batches to consume these data
                '''
                print('Enqueued new paths!')
                data_producer.enqueue_image_paths(sess, 'train')
                print('running new epoch!')
                for batch in range(cfg.TRAIN.BATCH_NUM):
                    # image_batchx, label_batchx = sess.run([image_batch, label_batch])
                    timer.tic()
                    train_image_batch, train_label_batch = data_producer.get_batch_data(sess, 'train')
                    total_loss_value, _ = sess.run([total_loss, train_op], feed_dict={self.net.data: train_image_batch, self.net.keep_prob:0.4, label_placeholder: train_label_batch})
                    timer.toc()

                    if (iter+1) % cfg.TEST.ITERS == 0:
                        # test the model
                        print('test now!')
                        if test_epoch % cfg.TEST.BATCH_NUM == 0:
                            data_producer.enqueue_image_paths(sess, 'test')
                        test_epoch = test_epoch + 1
                        test_image_batch, test_label_batch = data_producer.get_batch_data(sess, 'test')
                        test_accuracy = sess.run(accuracy, feed_dict={self.net.data: test_image_batch, self.net.keep_prob:1, label_placeholder: test_label_batch})
                        print("test accuracy: %.2f" %(test_accuracy))

                    print("epoch: %d/ %d, batch: %d/ %d, total loss: %.4f" %(epoch+1, cfg.TRAIN.NUM_EPOCHS, batch+1, cfg.TRAIN.BATCH_NUM, total_loss_value))
                    print("speed: %.3f" %(timer.average_time))
                   
                    if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        self.snapshot(sess, iter)
              
                    iter = iter + 1

    def train_vgg16_model(self):
        # we only allow 80% usage of gpu
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        coord = tf.train.Coordinator()
        
        
        with sess.as_default():
            # Three solftmax layers
            
            # start the data producer
            data_producer = DataProducer(self.train_annotation_filename, self.test_annotation_filename)
            
            data_producer.setup(sess, coord)
            
            image_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS))

            label_placeholder = tf.placeholder(tf.int64, shape=(None, ))

            self.net = VGG16_Train()
            
            loss_logits = self.net.get_output('fc8_new')

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=label_placeholder, logits=loss_logits, name='cross_entropy_per_example')
            
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            total_loss = cross_entropy_mean

            # prediction accuracy
            pred_prob = self.net.get_output('prob')
            correct_prediction = tf.equal(tf.argmax(pred_prob, 1), label_placeholder)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, cfg.TRAIN.STEP_SIZE, 0.1, staircase=True)

            momemtum = cfg.TRAIN.MOMENTUM

            train_op = tf.train.MomentumOptimizer(lr, momemtum).minimize(total_loss, global_step=global_step)
            
            # initialize variables
            sess.run(tf.global_variables_initializer())
            
            
            if self.pretrained_model is not None:
                print ('Loading pretrained model '
                        'weights from {:s}').format(self.pretrained_model)
                # note that we modify the structure of googlenet
                # concretely, the terminal classifed classes are reset
                self.net.load(self.pretrained_model, sess)

            timer = Timer()

            self.saver = tf.train.Saver(max_to_keep=100)

            print('Training Executing!')
            
            
            iter = 0
            test_epoch = 0
            print('Go to epoch!')
            for epoch in range(cfg.TRAIN.NUM_EPOCHS):
                '''
                At the begining of each epoch, all image paths should be enqueued
                Then we run serveral batches to consume these data
                '''
                print('Enqueued new paths!')
                data_producer.enqueue_image_paths(sess, 'train')
                print('running new epoch!')
                for batch in range(cfg.TRAIN.BATCH_NUM):
                    # image_batchx, label_batchx = sess.run([image_batch, label_batch])
                    timer.tic()
                    train_image_batch, train_label_batch = data_producer.get_batch_data(sess, 'train')
                    total_loss_value, _ = sess.run([total_loss, train_op], feed_dict={self.net.data: train_image_batch, self.net.keep_prob:0.5, label_placeholder: train_label_batch})
                    timer.toc()

                    if (iter+1) % cfg.TEST.ITERS == 0:
                        # test the model
                        print('test now!')
                        if test_epoch % cfg.TEST.BATCH_NUM == 0:
                            data_producer.enqueue_image_paths(sess, 'test')
                        test_epoch = test_epoch + 1
                        test_image_batch, test_label_batch = data_producer.get_batch_data(sess, 'test')
                        test_accuracy = sess.run(accuracy, feed_dict={self.net.data: test_image_batch, self.net.keep_prob:1, label_placeholder: test_label_batch})
                        print("test accuracy: %.2f" %(test_accuracy))

                    print("epoch: %d/ %d, batch: %d/ %d, total loss: %.4f" %(epoch+1, cfg.TRAIN.NUM_EPOCHS, batch+1, cfg.TRAIN.BATCH_NUM, total_loss_value))
                    print("speed: %.3f" %(timer.average_time))
                   
                    if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        self.snapshot(sess, iter)
              
                    iter = iter + 1


    def queue_test(self):
        sess = tf.Session()
        
        coord = tf.train.Coordinator()

        data_producer = DataProducer(self.train_annotation_filename, self.test_annotation_filename)

        data_producer.setup(sess, coord)

        with sess.as_default():
            data_producer.enqueue_image_paths(sess, 'train')
            for j in range(40):
                image_batch1, label_batch1 = data_producer.get_batch_data(sess, 'train')
                #image_batch, label_batch = sess.run(data_producer.get_batch_data(sess))
                print label_batch1
                print 'loading hahaha!!!!'
