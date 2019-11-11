import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from uci_adult import ADULT
from common.layers import xavier_init


class AAE:
    def __init__(
            self, input_size, n_z,
            architecture, learning_rate,
            activation_function=tf.nn.softplus):

        self.input_size = input_size
        self.n_z = n_z
        self.arch = architecture
        self.lr = learning_rate
        self.activation_function = activation_function

        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.z_real = tf.placeholder(tf.float32, [None, self.n_z])
        self.z = self.encoder()
        self.x_reconstr = self.decoder()

        prob_real = self.discriminator(self.z_real)
        prob_fake = self.discriminator(self.z)

        self.AE_loss = tf.reduce_mean(-tf.reduce_sum(self.x*tf.log(self.x_reconstr+1e-10)
                                                     + (1-self.x)*tf.log(1.-self.x_reconstr+1e-10), axis=1))
        self.D_loss = -tf.reduce_mean(tf.log(prob_real + 1e-30) + tf.log(1.0 - prob_fake + 1e-30))
        self.G_loss = -tf.reduce_mean(tf.log(prob_fake + 1e-30))

        self.autoEncoder_optimizer = tf.train.AdamOptimizer(self.lr['AE']).minimize(self.AE_loss)
        self.discriminator_optimizer = tf.train.AdamOptimizer(self.lr['D']).minimize(self.D_loss)
        self.generator_optimizer = tf.train.AdamOptimizer(self.lr['D']).minimize(self.G_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def encoder(self):
        w1_en = tf.Variable(xavier_init(self.input_size, self.arch['AE']['h1']))
        b1_en = tf.Variable(tf.zeros([1, self.arch['AE']['h1']]))
        h1_en = self.activation_function(tf.matmul(self.x, w1_en)+b1_en)

        w2_en = tf.Variable(xavier_init(self.arch['AE']['h1'], self.arch['AE']['h2']))
        b2_en = tf.Variable(tf.zeros([1, self.arch['AE']['h2']]))
        h2_en = self.activation_function(tf.matmul(h1_en, w2_en)+b2_en)

        w3_en = tf.Variable(xavier_init(self.arch['AE']['h2'], self.n_z))
        b3_en = tf.Variable(tf.zeros([1, self.n_z]))
        output = tf.matmul(h2_en, w3_en)+b3_en

        return output

    def decoder(self):
        w1_de = tf.Variable(xavier_init(self.n_z, self.arch['AE']['h2']))
        b1_de = tf.Variable(tf.zeros([1, self.arch['AE']['h2']]))
        h1_de = self.activation_function(tf.matmul(self.z, w1_de)+b1_de)

        w2_de = tf.Variable(xavier_init(self.arch['AE']['h2'], self.arch['AE']['h1']))
        b2_de = tf.Variable(tf.zeros([1, self.arch['AE']['h1']]))
        h2_de = self.activation_function(tf.matmul(h1_de, w2_de)+b2_de)

        w_reconstr = tf.Variable(xavier_init(self.arch['AE']['h1'], self.input_size))
        b_reconstr = tf.Variable(tf.zeros([1, self.input_size])+0.1)
        x_reconstr = tf.nn.sigmoid(tf.matmul(h2_de, w_reconstr)+b_reconstr)

        return x_reconstr

    def discriminator(self, z):
        w1 = tf.Variable(xavier_init(self.n_z, self.arch['D']['h1']))
        b1 = tf.Variable(tf.zeros([1, self.arch['D']['h1']]))
        h1 = self.activation_function(tf.matmul(z, w1)+b1)

        w2 = tf.Variable(xavier_init(self.arch['D']['h1'], 1))
        b2 = tf.Variable(tf.zeros([1, 1]))
        h2 = tf.nn.sigmoid(tf.matmul(h1, w2)+b2)

        return h2

    def fit_autoencoder(self, xs):
        _, cost = self.sess.run(
            (self.autoEncoder_optimizer, self.AE_loss),
            feed_dict={self.x: xs}
        )
        return cost

    def fit_discriminator(self, xs, zs):
        _, cost = self.sess.run(
            (self.discriminator_optimizer, self.D_loss),
            feed_dict={self.x: xs, self.z_real: zs}
        )
        return cost

    def fit_generator(self, xs):
        _, cost = self.sess.run(
            (self.generator_optimizer, self.G_loss),
            feed_dict={self.x: xs}
        )
        return cost

    def transform(self, xs):
        return self.sess.run(self.z, feed_dict={self.x: xs})

    def reconstruct(self, xs):
        return self.sess.run(self.x_reconstr, feed_dict={self.x: xs})

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def load_model(self, path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, path)


def train_aae(model, dataset, batch_size=128, epochs=100, display_step=0):
    print('Training AAE')
    for epoch in range(epochs):
        batch_xs, batch_ys = dataset.next_batch(batch_size)
        batch_zs = np.random.normal(0, 1, (batch_size, model.n_z))
        # n_z==2时，调整z的分布，配合draw_distribution函数查看效果：
        # batch_zs = np.vstack((np.random.normal(np.sin(batch_ys.argmax(1) * 36 * np.pi / 180), 0.16),
        #                       np.random.normal(np.cos(batch_ys.argmax(1) * 36 * np.pi / 180), 0.16))).T
        ae_loss = model.fit_autoencoder(batch_xs)
        d_loss = model.fit_discriminator(batch_xs, batch_zs)
        g_loss = model.fit_generator(batch_xs)
        if display_step > 0 and epoch % display_step == 0:
            print('Epoch: %5d/%d\tLoss: %f\t%f\t%f' % (epoch, epochs, ae_loss, d_loss, g_loss))


def draw_distribution(model, dataset):  # n_z==2时，可以显示10个label的分布图
    xs, ys = dataset.next_batch(1024, is_train=False)
    zs = model.transform(xs)
    color_set = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'grey', 'gold', 'goldenrod']
    plt.figure(figsize=(8, 8))
    for i in range(zs.shape[0]):
        plt.scatter(zs[i, 0], zs[i, 1], color=color_set[ys[i].argmax()])
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    adult = ADULT(path='../adult')
    aae = AAE(
        input_size=adult.n_features, n_z=64,
        architecture={
            'AE': {'h1': 512, 'h2': 256},
            'D': {'h1': 64}
        },
        learning_rate={'AE': 0.001, 'D': 0.001}
    )
    train_aae(aae, adult, batch_size=128, epochs=10000, display_step=100)
    draw_distribution(aae, adult)
