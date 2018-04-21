import tensorflow as tf
import os
from time import time

from common import cfg
import model, data_provider

eps = 1e-10


class Network(object):
    def __init__(self, args):# {{{
        self.args = args
        self.placeholder = {}
        self.net()
    # }}}
    def feed_dict(self, data):# {{{
        #   build feed_dict
        #   change dict{str->data} to dict{tensor->data}
        return {self.placeholder[key]: value \
                for key, value in data.items()}
    # }}}
    def show_epoch(self, loss, status):# {{{
        #   show training logs in shell
        current_time = time() - self.start_time
        eta = 'forever' if self.args.step == -1 else '%.2f' % \
                (current_time / self.step * (self.args.step - self.step))
        print('[LOG] #%d (%s): loss=%.4f time=%.2f eta=%s' % \
                (self.step, status, loss, current_time, eta))
    # }}}



    def func_result(self, x):# {{{
        '''
            input:
                x: output of net
            output:
                result (label, prediction or sth.) of final result
        '''
        with tf.name_scope('prediction'):
            _class = tf.argmax(x, axis=1)
            _score = tf.nn.softmax(x, name="softmax_tensor")
            return _class, _score
    # }}}
    def func_loss(self, x, y):# {{{
        '''
            input:
                x: output of net
                y: ground truth
            output:
                loss tensor (scalar)
        '''
        with tf.name_scope('loss'):
            x = tf.nn.softmax(x, axis=1, name='softmax')
            return -tf.reduce_mean(y*tf.log(x+eps) + (1-y)*tf.log(1-x+eps))
    # }}}
    def hidden_layer(self, x, is_training):# {{{
        '''
            hidden layer of the net
        '''
        with tf.name_scope('hidden_layer') as name_scope:
            return model.demo_model(x, is_training)
    # }}}
    def net(self):# {{{
        #   input layer
        with tf.name_scope('input_layer') as name:
            x = tf.placeholder('float', shape=[None, *cfg.input_shape], \
                    name='input_data')  #   input data
            y = tf.placeholder('float', shape=[None, *cfg.output_shape], \
                    name='label')       #   ground truth

            #   specify the status of net
            is_training = tf.placeholder(tf.bool, name='is_training')

            #   prepare for function feed_dict
            self.placeholder['input_data'] = x
            self.placeholder['label'] = y
            self.placeholder['is_training'] = is_training

        #   save input data in tensorboard
        tf.summary.image('input_data', x)   

        #   hidden layer
        x = self.hidden_layer(x, is_training)

        #   dense layer
        x = tf.layers.dense(inputs=x, units=cfg.output_shape[0], \
                name='logits')
        
        #   loss function
        self.loss_op = self.func_loss(x, y)
        
        #   optimizer & learning rate
        self.optimizer = tf.train.AdamOptimizer(\
                learning_rate=self.args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        
        #   prediction result
        self.predict_op = self.func_result(x)

        #   save loss for tensorboard
        tf.summary.scalar('loss', self.loss_op)

        #   merge all summary for tensorboard
        self.summary = tf.summary.merge_all()
    # }}}

    def train(self):# {{{
        #   initializer all variables
        init = tf.global_variables_initializer()

        #   saver to save/restore model to/from path
        saver = tf.train.Saver()

        #   tf session
        with tf.Session() as sess:

            #   run initalizer for all variables
            sess.run(init)

            #   load pre-trained model if needed
            if self.args.checkpoint:
                print('[LOG] loading model from', self.args.checkpoint)
                saver.restore(sess, self.args.checkpoint)

            #   train/val writer of tensorboard
            train_writer = tf.summary.FileWriter(\
                    os.path.join(self.args.logdir, 'train'), sess.graph)
            val_writer = tf.summary.FileWriter(\
                    os.path.join(self.args.logdir, 'val'))
            
            print('[LOG] training started ..')
            self.step, self.start_time = 0, time()
            while self.step < self.args.step or self.args.step == -1:
                self.step += 1

                #   get train data
                train_batch = data_provider.get_train(self.args.batch_size)

                #   get training summary, loss value, 
                #   and parameters updating
                train_summary, loss, _ = sess.run(
                    [self.summary, self.loss_op, self.train_op], 
                    feed_dict=self.feed_dict(train_batch)
                )

                #   recorde summary of each step for tensorboard
                train_writer.add_summary(train_summary, self.step)

                #   save model
                if self.step % self.args.save_step == 0:
                    name = 'model_%d.pkl' % self.step
                    saver.save(sess, \
                            os.path.join(self.args.model_path, name))

                #   display train/val loss
                if self.step == 1 or \
                        self.step % self.args.val_step == 0:
                    self.show_epoch(loss, 'train')

                    val_batch = data_provider.get_val(self.args.batch_size)
                    val_summary, loss = sess.run(\
                            [self.summary, self.loss_op], \
                            feed_dict=self.feed_dict(val_batch))
                    val_writer.add_summary(val_summary, self.step)
                    self.show_epoch(loss, 'validation')
        print('[LOG] Training finished ..')
            
    # }}}
