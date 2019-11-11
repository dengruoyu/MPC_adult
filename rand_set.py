import numpy as np


class RandSet:
    def __init__(self, n_samples_train, n_samples_test, n_features, n_labels):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_features = n_features
        self.n_labels = n_labels

    def next_batch(self, batch_size=128, is_train=True):
        batch_xs = np.random.randint(0, 2, (batch_size, self.n_features))
        batch_ys = np.eye(self.n_labels)[np.random.randint(0, 2, batch_size)]
        return batch_xs, batch_ys


if __name__ == '__main__':
    pass
