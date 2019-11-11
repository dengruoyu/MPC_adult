import numpy as np
import tensorflow as tf
from common.layers import dense
from common.utils import cal_acc


class DNN:
    def __init__(
            self, input_size, output_size, architecture,
            learning_rate=0.001, lam=0.005,
            activation_function=tf.nn.softplus):

        self.input_size = input_size
        self.output_size = output_size
        self.arch = architecture
        self.lr = learning_rate
        self.activation_function = activation_function

        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])
        self.y_pre, norm_loss = self.create_network()

        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pre + 1e-10), axis=1))

        self.loss = cross_entropy_loss + lam * norm_loss
        # self.loss = cross_entropy_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def create_network(self):
        h1, w1, b1 = dense(
            self.x, self.input_size, self.arch['h1'],
            self.activation_function, return_param=True
        )
        h2, w2, b2 = dense(
            h1, self.arch['h1'], self.arch['h2'],
            self.activation_function, return_param=True
        )
        output, w3, b3 = dense(
            h2, self.arch['h2'], self.output_size,
            tf.nn.softmax, return_param=True
        )

        norm_loss = tf.reduce_sum(tf.square(w1))\
                    + tf.reduce_sum(tf.square(b1))\
                    + tf.reduce_sum(tf.square(w2))\
                    + tf.reduce_sum(tf.square(b2))\
                    + tf.reduce_sum(tf.square(w3))\
                    + tf.reduce_sum(tf.square(b3))
        return output, norm_loss

    def partial_fit(self, xs, ys):
        _, cost = self.sess.run(
            (self.optimizer, self.loss),
            feed_dict={self.x: xs, self.y: ys}
        )
        return cost

    def predict(self, xs):
        return self.sess.run(self.y_pre, feed_dict={self.x: xs})

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def load_model(self, path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, path)


def train_dnn(model, users, batch_size=128, epochs=100, display_step=0):
    for epoch in range(epochs):
        dataset = users[np.random.randint(len(users))]
        batch_xs, batch_ys = dataset.next_batch(batch_size=128)
        loss = model.partial_fit(batch_xs, batch_ys)
        if display_step > 0 and epoch % display_step == 0:
            acc_train = cal_acc(model.predict(batch_xs), batch_ys)
            batch_xs, batch_ys = dataset.next_batch(batch_size, is_train=False)
            acc_test = cal_acc(model.predict(batch_xs), batch_ys)
            print('Epoch: %5d/%d\tLoss: %f\tAcc: %f\t %f' %
                  (epoch, epochs, loss, acc_train, acc_test))


if __name__ == '__main__':
    pass
