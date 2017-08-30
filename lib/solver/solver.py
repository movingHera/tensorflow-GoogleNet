from config.config import cfg
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import time
from dataset.dataset import DataProducer
from networks.GoogLeNet_Train import GoogLeNet_Train
from networks.VGG16_Train import VGG16_Train

class SolverWrapper(object):
    '''A solver to control the training process and execute snapshotting process
    
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

    def average_gradients(self, tower_grads):
        '''Average gradients for variables in tower_grads

        Args:
            tower_grads: [((grad0_gpu0, var0_gpu0), ... (gradN_gpu0,
            varN_gpu0)), ((grad0_gpu1, var0_gpu1), ..., (gradN_gpu1,
            varN_gpu1)), ...]
        
        Returns:
            average_grads: [(grad0, var0), ..., (gradN, varN)]
        '''
        
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Transform tower_grads to the format: 
            # [((grad0_gpu0, var0_gpu0), ...(grad0_gpuM, var0_gpuM)), ...,
            # (gradN_gpu0, varN_gpu0), ...(gradN_gpuM, varN_gpuM))]
            grads = [g for g, _ in grad_and_vars]
            # Average over the 'tower' dimension
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Varaibles are redundant because they are
            # shared across towers. So .. we will just return the first towers'
            # pointer to the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def feed_all_gpu(self, inp_dict, models, payload_per_gpu, batch_x, batch_y,
            keep_prob):
        '''Construct feed_dict for each gpu

        Args:
            inp_dict: dictionary for input
            models: the first 3 elements are placeholder for inputs and
                    labels and keep prob, 4-6th elements are predicts, loss, grads
            payload_per_gpu: the number of inputs for each gpu
            batch_x, batch_y: a large batch of data, which will be distributed
                              into each gpu
        Returns:
            inp_dict[x] and inp_dict[y] for each gpu
        '''
        for i in range(len(models)):
            x, y, kp, _, _, _ = models[i]
            start_pos = i * payload_per_gpu
            end_pos = (i+1) * payload_per_gpu
            inp_dict[x] = batch_x[start_pos: end_pos]
            inp_dict[y] = batch_y[start_pos: end_pos]
            inp_dict[kp] = keep_prob

        return inp_dict

    def train_googlenet_multigpu(self):
        '''Train googlenet model in multi-gpu environment

        '''
        ngpus = 4  # Currently we have 4-gpus machine

        # The load for each gpu
        payload_per_gpu = int(cfg.TRAIN.BATCH_SIZE / ngpus)
        
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        coord = tf.train.Coordinator()

        # Producer for training/test data
        train_data_producer = DataProducer(cfg.DATASET_NAME, 'train',
                payload_per_gpu * ngpus)
        test_data_producer = DataProducer(cfg.DATASET_NAME, 'test')

        train_data_producer.setup(sess, coord)
        test_data_producer.setup(sess, coord)

        # Variable lists that have been trained on ImageNet
        pretrained_list = []
        last_var = []

        global_step = tf.Variable(0, trainable=False)

        # set 10x learning rate for the last fc layer
        lr_small = \
            tf.train.exponential_decay(cfg.TRAIN.BASE_LERANING_RATE,
                global_step, cfg.TRAIN.STEP_SIZE, 0.1,
                staircase=True)

        lr_large = \
            tf.train.exponential_decay(10*cfg.TRAIN.BASE_LERANING_RATE,
                global_step, cfg.TRAIN.STEP_SIZE, 0.1,
                staircase=True)
       
        # opt1 for pretrained variables
        # opt2 for the last fc layer
        opt1 = tf.train.MomentumOptimizer(lr_small,
                cfg.TRAIN.MOMENTUM)
        opt2 = tf.train.MomentumOptimizer(lr_large,
                cfg.TRAIN.MOMENTUM)

        with sess.as_default(), tf.device('/cpu:0'):
            models = []
            for gpu_id in range(ngpus):
                with tf.device('/gpu: %d' % gpu_id):
                    with tf.name_scope('tower_%d'  % gpu_id):
                        with tf.variable_scope("", reuse=gpu_id>0):
                            y = tf.placeholder(tf.int64,
                                    shape=(None, ))
                           
                            net = GoogLeNet_Train()
                            
                            if gpu_id == 0:
                                self.__net = net
                                pretrained_list = net.pretrained_variable_list
                                last_var = \
                                    net.get_params('loss3_classifier_new',
                                        'weights, biases')
                                net.load(self.pretrained_model, sess)
                            
                            probs = net.get_output('prob3')
                           
                            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=y,
                                    logits=logits,
                                    name='cross_entropy_per_example')   
                            loss = tf.reduce_mean(cross_entropy,
                                    name='cross_entropy')
                            
                            grads = tf.gradients(loss)
                            models.append((net.data, y, net.keep_prob, probs, loss, grads))

            tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)

            avg_loss_op = tf.reduce_mean(tower_losses)
           
            # Group two ops into one op
            tower_grads1 = tower_grads[:len(pretrained_list)]
            tower_grads2 = tower_grads[len(pretrained_list):]
            train_op1 = opt1.apply_gradients(zip(grads1,
                                pretrained_list), global_step=global_step)
            train_op2 = opt2.compute_gradients(zip(grads2,
                                last_var), global_step=global_step)
            apply_gradient_op = tf.group(train_op1, train_op2)
            
            # For test
            all_y = tf.stack(tower_y, 0)
            all_preds = tf.stack(tower_preds, 0)
            correct_pred = tf.equal(tf.argmax(all_preds, 1), all_y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

            
            sess.run(tf.variables_initializer(set(tf.all_variables()) -
                set(pretrained_list)))

            print('Finish initialization!')
            for epoch in range(cfg.NUM_EPOCHS):
                train_data_num = train_data_producer.get_data_num()
                batch_num = int(np.ceil(train_data_num / payload_per_gpu *
                    ngpus))
                
                for batch in range(batch_num):
                    train_image_batch, train_label_batch = \
                        train_data_producer.get_batch_data(sess)
                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict = self.feed_all_gpu(inp_dict, models, payload_per_gpu,
                            train_image_batch, train_label_batch, 0.5)
                    _, _loss = sess.run([apply_gradient_op, avg_loss_op], inp_dict)

                    print('Train loss is: %.4f' %(_loss))

                if (epoch+1) % cfg.TEST.EPOCHS == 0:
                    preds = None
                    ys = None

                    ''' The total test data is more than those used in test
                        Because we test num must be divided by payload_per_gpu * ngpus
                        Thus some data will be discarded
                    '''
                    test_data_num = test_data_producer.get_batch_data(sess)
                    batch_num = int(np.ceil(test_data_num / (payload_per_gpu *
                        ngpus)))

                    test_image_batch, test_label_batch = \
                            test_data_producer.get_batch_data(sess)
                    
                    preds = None
                    ys = None
                    for batch_idx in range(batch_num):
                        start_pos = batch_idx * payload_per_gpu * ngpus
                        end_pos = (batch_idx+1) * payload_per_gpu * ngpus
                        inp_dict = {}
                        inp_dict = self.feed_all_gpu(inp_dict, models,
                                payload_per_gpu, test_image_batch[start_pos:
                                    end_pos], test_label_batch[start_pos:
                                        end_pos], 1)
                        batch_pred, batch_y = sess.run([all_pred, all_y], \
                                              inp_dict)
                        
                        if preds is None:
                            preds = batch_pred
                        else:
                            preds = np.concatenate((preds, batch_pred), 0)

                        if ys is None:
                            ys = batch_y
                        else:
                            ys = np.concatenate((ys, batch_y), 0)

                    val_accuracy = sess.run([accuracy], {all_y:ys,
                        all_pred:preds})[0]
                    print "**************************"
                    print('Val accuracy: %0.4f%%' %(100 * val_accuracy))
                    print "**************************"

                if (epoch+1) % cfg.TRAIN.SNAPSHOT_EPOCHS:
                    self.snapshot(sess, epoch)


    def train_googlenet_model(self):
        # we only allow 80% usage of gpu
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        coord = tf.train.Coordinator()
        
        
        with sess.as_default():
            
            # Construct the training/test data producer
            train_data_producer = DataProducer(cfg.DATASET_NAME, 'train', cfg.TRAIN.BATCH_SIZE)
            test_data_producer = DataProducer(cfg.DATASET_NAME, 'test')

            train_data_producer.setup(sess, coord)
            test_data_producer.setup(sess, coord)

            image_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CROP_SIZE, cfg.PREPROCESS.CHANNELS))

            label_placeholder = tf.placeholder(tf.int64, shape=(None, ))

            self.__net = GoogLeNet_Train()
            
            #loss1_logits = self.net.get_output('loss1_classifier')
            #loss2_logits = self.net.get_output('loss2_classifier')
            loss3_logits = self.__net.get_output('loss3_classifier_new')

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
          
            self.saver = tf.train.Saver(max_to_keep=100)
           
            # set 10x learning rate for the last fc layer
            # pretrained_var: the params that have been trained on imagenet
            # last_var: the params in last fc layer
            pretrained_var = self.__net.get_pretrained_variable_list
            last_var = self.__net.get_params('loss3_classifier_new', 'weights',
                    'biases')

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
                self.__net.load(self.pretrained_model, sess)

            print('Training Executing!')
            
            print('Go to epoch!')
            for epoch in range(cfg.TRAIN.NUM_EPOCHS):
                train_data_num = train_data_producer.get_data_num()
                batch_num = int(np.ceil(train_data_num / cfg.TRAIN.BATCH_SIZE))
                for batch in range(batch_num):
                    # image_batchx, label_batchx = sess.run([image_batch, label_batch])
                    timer.tic()
                    train_image_batch, train_label_batch = train_data_producer.get_batch_data(sess)
                    total_loss_value, _ = sess.run([total_loss, train_op], feed_dict={self.net.data: train_image_batch, self.net.keep_prob:0.4, label_placeholder: train_label_batch})
                    timer.toc()
                    
                    print("epoch: %d/ %d, batch: %d/ %d, total loss: %.4f" %(epoch+1, cfg.TRAIN.NUM_EPOCHS, batch+1, cfg.TRAIN.BATCH_NUM, total_loss_value))
                    print("speed: %.3f" %(timer.average_time))

                    if (epoch+1) % cfg.TEST.EPOCHS == 0:
                        # test the model
                        print('test now!')
                        test_image_batch, test_label_batch = test_data_producer.get_batch_data(sess)
                        test_data_num = test_data_producer.get_data_num()

                        # Test mini_batch data each time
                        mini_batch = 200
                        mini_batch_num = int(np.ceil(test_data_num / mini_batch)) + 1
                        
                        all_prediction = np.zeros(test_data_num, dtype=bool)
                        for batch_idx in range(mini_batch_num):
                            start_idx = batch_idx * mini_batch
                            end_idx = min((batch_idx+1) * mini_batch,
                                    test_data_num)
                            current_image_batch = test_image_batch[start_idx:
                                    end_idx,: ]
                            current_label_batch = test_label_batch[start_idx:
                                    end_idx, :]
                            current_pred = sess.run(correct_prediction, feed_dict=
                                    {self.net.data: test_image_batch, self.net.keep_prob:1, 
                                        label_placeholder: test_label_batch})
                            all_prediction[start_idx: end_idx] = current_pred
                                    
                        all_prediction.astype(np.float32)
                        print "*******************************"
                        print("test accuracy: %.2f" %(np.mean(all_prediction)))
                        print "*******************************"

                    if (epoch+1) % cfg.TRAIN.SNAPSHOT_EPOCHS == 0:
                        self.snapshot(sess, epoch)

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
