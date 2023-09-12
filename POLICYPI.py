
import numpy as np
import tensorflow as tf

class Policy(object):
    """ RED NEURONAL DE LA POLICY """
    def __init__(self, obs_dim, act_dim, epsilon, mult_neuronas, epoch, policy_std):
        tf.set_random_seed(2)
        self.mult_neuronas = mult_neuronas
        self.epochs = epoch
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epsilon = epsilon
        self.policy_std = policy_std

        """ PLACEHOLDERS """
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.old_logstd_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_logstd')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

        """ RED NEURONAL PARA LA APROXIMACION DE LA POLICY PARAMETRIZADA POR
        UNA GAUSSEANA USANDO MEAN Y VARIANCES PARA DETERMINAR LAS ACCIONES """
        hid1_size = self.obs_dim * self.mult_neuronas
        hid3_size = self.act_dim * self.mult_neuronas
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = 9e-5 / np.sqrt(hid2_size)
        print('Policy Net: \nLayer1 size: ', hid1_size, '\nLayer2 size: ', hid2_size, '\nLayer3 size: ',hid3_size,
              '\nLearning Rate: ', self.lr)
        # NN 3 CAPAS CON ACT FUNCT TANH
        with tf.variable_scope('policy_nn'):
            self.layer1 = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / obs_dim)),name="h1_policy")
            self.layer2 = tf.layers.dense(self.layer1, hid2_size, tf.tanh,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid1_size)), name="h2_policy")
            self.layer3 = tf.layers.dense(self.layer2, hid3_size,tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid2_size)), name="h3_policy")
            self.means =  tf.layers.dense(self.layer3, self.act_dim,kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid3_size)),name="means")
            # log_var_speed se usa para hacer actualizaciones mas rapido
            logstd_speed = (10 * hid3_size) // 48
            log_std = tf.get_variable('log_stds', (logstd_speed, self.act_dim), tf.float32, tf.zeros_initializer())
            self.log_std = tf.reduce_sum(log_std, axis=0) + self.policy_std

        """ CALCULA LOS PARAMETROS: """
        self._logprob()
        self._sample()
        self._loss_train_op()

        """ INICIO DE SESSION """
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.merged = tf.summary.merge([self.logp_resta_sum, self.loss_sum, self.adv_sum, self.ratios_sum])


    def _logprob(self):
        """ CALCULA LOG PROBABILITY DE POLICY PI """
        logp = 0.5 * tf.reduce_sum(tf.square((self.act_ph - self.means) / tf.exp(self.log_std)), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.act_ph)[-1]) \
               + tf.reduce_sum(self.log_std, axis=-1)
        self.logp = -logp

        old_logp = 0.5 * tf.reduce_sum(tf.square((self.act_ph - self.old_means_ph) / tf.exp(self.old_logstd_ph)), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.act_ph)[-1]) \
               + tf.reduce_sum(self.old_logstd_ph, axis=-1)
        self.old_logp = -old_logp

    def _loss_train_op(self):

        """ CLIPPED LOSS """
        with tf.variable_scope('loss/clip'):
            ratios = tf.exp(self.logp - self.old_logp)
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)
            loss_clip = tf.minimum(tf.multiply(self.advantages_ph, ratios), tf.multiply(self.advantages_ph, clipped_ratios))
            clip_loss = -tf.reduce_mean(loss_clip)

            self.logp_resta_sum = tf.summary.scalar('Resta de Logp', tf.reduce_mean(self.logp - self.old_logp))
            self.ratios_sum = tf.summary.scalar('Ratio', tf.reduce_mean(ratios))
            self.adv_sum = tf.summary.scalar('Advantages', tf.reduce_mean(self.advantages_ph))

        """ TOTAL POLICY LOSS """
        with tf.variable_scope('loss/total_loss'):
            self.policy_loss = clip_loss

            self.loss_sum = tf.summary.scalar('Total Loss', self.policy_loss)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.policy_loss)


    def _sample(self):
        """ CALCULA UNA ACCION DE MUESTRA CON DISTRIBUCION GAUSSEANA """
        self.sampled_act = self.means + tf.exp(self.log_std) * tf.random_normal(shape=(self.act_dim,))


    def sample(self, obs):
        """ ENTREGA UNA ACCION DE LA POLICY DADA LA OBS"""
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """ ACTUALIZA LA POLICY """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages}
        old_means_np, old_log_std_np = self.sess.run([self.means, self.log_std],feed_dict)
        feed_dict[self.old_logstd_ph] = old_log_std_np
        feed_dict[self.old_means_ph] = old_means_np

        for e in range(self.epochs):
            self.sess.run([self.train_op, self.policy_loss], feed_dict)
        return self.sess.run(self.merged, feed_dict)

    def get_summary(self, obs, act, adv):
        feed_dict = {self.obs_ph: obs,
                     self.act_ph: act,
                     self.advantages_ph: adv}
        old_means_np, old_log_std_np = self.sess.run([self.means, self.log_std], feed_dict)
        feed_dict[self.old_means_ph] = old_means_np
        feed_dict[self.old_logstd_ph] = old_log_std_np
        return self.sess.run(self.merged, feed_dict=feed_dict)

    def close_sess(self):
        self.sess.close()
