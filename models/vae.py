import tensorflow as tf
import matplotlib.pyplot as plt

from uci_adult import ADULT
from common.layers import xavier_init


class VAE:
    def __init__(
            self, input_size, architecture,
            learning_rate=0.001,
            activation_function=tf.nn.softplus):

        self.input_size = input_size
        self.arch = architecture
        self.lr = learning_rate
        self.activation_function = activation_function

        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.z_mean, self.z_log_sigma_sq = self.encoder()
        eps = tf.random_normal([self.arch['z']], 0, 1)
        self.z = self.z_mean + eps*tf.sqrt(tf.exp(self.z_log_sigma_sq))
        self.x_reconstr = self.decoder()

        reconstr_loss = -tf.reduce_sum(self.x*tf.log(self.x_reconstr+1e-10)
                                       + (1-self.x)*tf.log(1-self.x_reconstr+1e-10),
                                       axis=1)
        latent_loss = -0.5*tf.reduce_sum(1 + self.z_log_sigma_sq
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma_sq),
                                         axis=1)
        self.loss = tf.reduce_mean(reconstr_loss+latent_loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def encoder(self):
        w1_en = tf.Variable(xavier_init(self.input_size, self.arch['h1']))
        b1_en = tf.Variable(tf.zeros([1, self.arch['h1']]))
        h1_en = self.activation_function(tf.matmul(self.x, w1_en)+b1_en)

        w2_en = tf.Variable(xavier_init(self.arch['h1'], self.arch['h2']))
        b2_en = tf.Variable(tf.zeros([1, self.arch['h2']]))
        h2_en = self.activation_function(tf.matmul(h1_en, w2_en)+b2_en)

        w_mean = tf.Variable(xavier_init(self.arch['h2'], self.arch['z']))
        b_mean = tf.Variable(tf.zeros([1, self.arch['z']]))
        z_mean = tf.matmul(h2_en, w_mean)+b_mean

        w_log_sigma_sq = tf.Variable(xavier_init(self.arch['h2'], self.arch['z']))
        b_log_sigma_sq = tf.Variable(tf.zeros([1, self.arch['z']]))
        z_log_sigma_sq = tf.matmul(h2_en, w_log_sigma_sq)+b_log_sigma_sq

        return z_mean, z_log_sigma_sq

    def decoder(self):
        w1_de = tf.Variable(xavier_init(self.arch['z'], self.arch['h2']))
        b1_de = tf.Variable(tf.zeros([1, self.arch['h2']]))
        h1_de = self.activation_function(tf.matmul(self.z, w1_de)+b1_de)

        w2_de = tf.Variable(xavier_init(self.arch['h2'], self.arch['h1']))
        b2_de = tf.Variable(tf.zeros([1, self.arch['h1']]))
        h2_de = self.activation_function(tf.matmul(h1_de, w2_de)+b2_de)

        w_reconstr = tf.Variable(xavier_init(self.arch['h1'], self.input_size))
        b_reconstr = tf.Variable(tf.zeros([1, self.input_size])+0.1)
        x_reconstr = tf.nn.sigmoid(tf.matmul(h2_de, w_reconstr)+b_reconstr)

        return x_reconstr

    def partial_fit(self, xs):
        _, cost = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: xs})
        return cost

    def cal_cost(self, xs):
        cost = self.sess.run(self.loss, feed_dict={self.x: xs})
        return cost

    def transform(self, xs):
        return self.sess.run(self.z_mean, feed_dict={self.x: xs})

    def reconstruct(self, xs):
        return self.sess.run(self.x_reconstr, feed_dict={self.x: xs})

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def load_model(self, path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, path)


def train_vae(model, dataset, batch_size=128, epochs=100, display_step=0):
    print('Training VAE')
    for epoch in range(epochs):
        batch_xs, _ = dataset.next_batch(batch_size)
        loss = model.partial_fit(batch_xs)
        if display_step > 0 and epoch % display_step == 0:
            batch_xs, _ = dataset.next_batch(batch_size, is_train=False)
            test_loss = model.cal_cost(batch_xs)
            print('Epoch: %5d/%d\tCost: %f\t %f' %
                  (epoch, epochs, loss, test_loss))


def draw_distribution(model, dataset):
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
    vae = VAE(
        input_size=adult.n_features,
        architecture={'h1': 128, 'h2': 128, 'z': 64},
        learning_rate=0.001
    )
    train_vae(vae, adult, batch_size=128, epochs=100000, display_step=100)
    draw_distribution(vae, adult)
