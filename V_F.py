import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys

class NNValueFunction(object):
    """ RED NEURONAL DE VALUE FUNCTION """
    def __init__(self, obs_dim, epsilon, mult_neuronas, epochs):
        """ PLACEHOLDERS Y PARAMETROS """
        tf.set_random_seed(2)
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.epochs = epochs
        self.mult_neuronas = mult_neuronas

        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
        self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')

        """ RED NEURONAL DE APROXIMACION DE VF """
        hid1_size = self.obs_dim * self.mult_neuronas
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = 1e-3 / np.sqrt(hid2_size)
        print('Value Function Net: \n Layer1 size: ', hid1_size,'\nLayer2 size: ', hid2_size, '\nLayer3 size: ', hid3_size,
              '\nLearning Rate: ', self.lr, '\n')

        # NN 3 CAPAS CON ACT FUNCT TANH
        layer1 = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name="h1_vf")
        layer2 = tf.layers.dense(layer1, hid2_size, tf.tanh,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)), name="h2_vf")
        layer3 = tf.layers.dense(layer2, 5, tf.tanh,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid2_size)), name="h3_vf")
        out = tf.layers.dense(layer3, 1,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid3_size)), name='output')
        self.out = tf.squeeze(out)

        """" VF LOSS """
        with tf.variable_scope('loss/vf_loss'):
            #self.vf_loss1 = tf.square(self.out - self.val_ph)
            #vpredclipped = self.old_val_ph + tf.clip_by_value(self.out - self.old_val_ph, - self.epsilon, self.epsilon)
            #self.vf_loss2 = tf.square(vpredclipped - self.val_ph)
            #self.vf_loss = .5 * tf.reduce_mean(tf.maximum(self.vf_loss1, self.vf_loss2))
            self.vf_loss = .5 * tf.reduce_mean(tf.square(self.out - self.val_ph))

            optimizer = tf.train.AdamOptimizer(self.lr)

            self.train_op = optimizer.minimize(self.vf_loss)
            self.vf_loss_sum = tf.summary.scalar('VF Loss', self.vf_loss)
        
        """ INICIO DE SESSION """
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        #self.merged = tf.summary.merge([self.vf_loss_sum])

    def fit(self, x, y):
        """ x: observaciones, y: real value function """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y

        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                self.sess.run([self.train_op, self.vf_loss], feed_dict=feed_dict)

    def predict(self, obs):
        """ ENTREGA VF APROXIMADO POR LA RED """
        feed_dict = {self.obs_ph: obs}
        v_pred = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(v_pred)

    def get_summary(self, obs, val):
        feed_dict = {self.obs_ph: obs,
                     self.val_ph: val}
        return self.sess.run(self.merged, feed_dict=feed_dict)

    def close_sess(self):
        self.sess.close()
