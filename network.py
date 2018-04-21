import tensorflow as tf
import os
from IPython import embed
from time import time

from common import cfg
from data_provider import *
import model



class Network(object):
    def __init__(self, args):# {{{
        self.args = args
        self.placeholder = {}
        self.net()
    # }}}
    def feed_dict(self, data):# {{{
        return {self.placeholder[key]: value for key, value in data.items()}
    # }}}
    def show_epoch(self, loss, name):# {{{
        current_time = time() - self.start_time
        eta = 'forever' if self.args.step == -1 else '%.2f' % \
                (current_time / self.step * (self.args.step - self.step))
        print('[%s] #%d: loss=%.4f time=%.2f eta=%s' % \
                (name, self.step, loss, current_time, eta))
    # }}}



    def func_result(self, x):# {{{
        with tf.name_scope('prediction'):
            _class = tf.argmax(x, axis=1)
            _score = tf.nn.softmax(x, name="softmax_tensor")
            return _class, _score
    # }}}
    def func_loss(self, x, y):# {{{
        with tf.name_scope('loss'):
            x = tf.nn.softmax(x, axis=1, name='softmax')
            return -tf.reduce_mean(y*tf.log(x) + (1-y)*tf.log(1-x))
    # }}}
    def hidden_layer(self, x, is_training):# {{{
        with tf.name_scope('hidden_layer') as name_scope:
            return model.demo_model(x, is_training)
    # }}}
    def net(self):# {{{
        with tf.name_scope('input_layer') as name:
            x = tf.placeholder('float', shape=[None, *cfg.input_shape], \
                    name='input_data')
            y = tf.placeholder('float', shape=[None, *cfg.output_shape], \
                    name='label')
            is_training = tf.placeholder(tf.bool, name='is_training')
            self.placeholder['input_data'] = x
            self.placeholder['label'] = y
            self.placeholder['is_training'] = is_training
        tf.summary.image('input_data', x)

        x = self.hidden_layer(x, is_training)
        x = tf.layers.dense(inputs=x, units=cfg.output_shape[0], \
                name='logits')
        
        self.loss_op = self.func_loss(x, y)
        self.optimizer = tf.train.AdamOptimizer(\
                learning_rate=self.args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.predict_op = self.func_result(x)

        tf.summary.scalar('loss', self.loss_op)
        self.summary = tf.summary.merge_all()
    # }}}
    def train(self):# {{{
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            if self.args.checkpoint:
                saver.restore(sess, self.args.checkpoint)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(\
                    os.path.join(self.args.logdir, 'train'), sess.graph)
            val_writer = tf.summary.FileWriter(\
                    os.path.join(self.args.logdir, 'val'))

            self.step, self.start_time = 0, time()
            while self.step < self.args.step or self.args.step == -1:
                self.step += 1
                train_batch = get_train(self.args.batch_size)
                train_summary, loss, _ = sess.run(
                    [self.summary, self.loss_op, self.train_op], 
                    feed_dict=self.feed_dict(train_batch)
                )
                train_writer.add_summary(train_summary, self.step)

                if self.step % self.args.save_step == 0:
                    saver.save(sess, '%s_%d.pkl' % \
                            (os.path.join(self.args.model_path, \
                            self.args.name), self.step))

                if self.step == 1 or \
                        self.step % self.args.val_step == 0:
                    self.show_epoch(loss, 'train')
                    val_summary, loss = sess.run(\
                            [self.summary, self.loss_op], \
                            feed_dict=self.feed_dict(get_val(\
                            self.args.batch_size)))
                    val_writer.add_summary(val_summary, self.step)
                    self.show_epoch(loss, 'validation')
            
    # }}}
