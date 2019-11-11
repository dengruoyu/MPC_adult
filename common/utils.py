import numpy as np


def split_dims(ori_dim_list, num_list, in_order=False):
    dim_list = []
    for i in num_list:
        if in_order:
            dims = np.arange(i)
        else:
            dims = np.random.choice(ori_dim_list.shape[0], i, replace=False)
        dim_list.append(ori_dim_list[dims])
        ori_dim_list = np.delete(ori_dim_list, dims)
    return dim_list


def random_batch(length, batch_size=128):
    return np.random.choice(length, batch_size, replace=False)


def cal_acc(y_pre, y):
    y_pre_choice = np.argmax(y_pre, axis=1)
    y_choice = np.argmax(y, axis=1)
    return np.mean(y_pre_choice == y_choice)


if __name__ == '__main__':
    pass
