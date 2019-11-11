import numpy as np

from uci_adult import ADULT
from models.vae import VAE, train_vae
from models.aae import AAE, train_aae
from models.dnn import DNN
from common.utils import cal_acc

n_users = 10
hidden_size = 64


def train_dnn(model, users, ae_list, user_id=None, batch_size=128, epochs=100, display_step=0):
    print('Training DNN')
    for epoch in range(epochs):
        if user_id is None:
            user_id = np.random.randint(n_users)
        dataset = users[user_id]
        ae = ae_list[user_id]

        batch_xs, batch_ys = dataset.next_batch(batch_size=128)
        loss = model.partial_fit(ae.transform(batch_xs), batch_ys)
        if display_step > 0 and epoch % display_step == 0:
            acc_train = cal_acc(model.predict(ae.transform(batch_xs)), batch_ys)
            batch_xs, batch_ys = dataset.next_batch(batch_size, is_train=False)
            acc_test = cal_acc(model.predict(ae.transform(batch_xs)), batch_ys)
            print('Epoch: %5d/%d\tLoss: %f\tAcc: %f\t %f' %
                  (epoch, epochs, loss, acc_train, acc_test))


if __name__ == '__main__':
    '''
    adult = ADULT()
    randset = RandSet(adult.n_samples_train, adult.n_samples_test,
                      adult.n_features, adult.n_labels)
    user_list = [adult, randset]
    '''
    user_list = [ADULT(n_users=n_users, user_id=i) for i in range(n_users)]
    ae_type = [1]*n_users

    ae_list = list()

    for i, user in enumerate(user_list):
        if ae_type[i] == 0:
            vae = VAE(
                input_size=user.n_features,
                architecture={'h1': 128, 'h2': 128, 'z': hidden_size},
                learning_rate=0.001
            )
            train_vae(vae, user, batch_size=128, epochs=20000, display_step=500)
            ae_list.append(vae)
        else:
            aae = AAE(
                input_size=user.n_features, n_z=hidden_size,
                architecture={
                    'AE': {'h1': 128, 'h2': 128},
                    'D': {'h1': 64}
                },
                learning_rate={'AE': 0.001, 'D': 0.001}
            )
            train_aae(aae, user, batch_size=128, epochs=10000, display_step=500)
            ae_list.append(aae)

    '''
    dnn_model_1 = DNN(
        input_size=hidden_size,
        output_size=user_list[0].n_labels,
        architecture={'h1': 128, 'h2': 128},
        learning_rate=0.001
    )
    train_dnn(dnn_model_1, user_list, ae_list, user_id=0, batch_size=128, epochs=1000, display_step=10)

    dnn_model_2 = DNN(
        input_size=hidden_size,
        output_size=user_list[0].n_labels,
        architecture={'h1': 128, 'h2': 128},
        learning_rate=0.001
    )
    train_dnn(dnn_model_2, user_list, ae_list, user_id=1, batch_size=128, epochs=1000, display_step=10)
    '''

    dnn_model = DNN(
        input_size=hidden_size,
        output_size=user_list[0].n_labels,
        architecture={'h1': 128, 'h2': 128},
        learning_rate=0.001
    )
    train_dnn(dnn_model, user_list, ae_list, batch_size=128, epochs=10000, display_step=100)

    batch_xs, batch_ys = user_list[0].next_batch(user_list[0].n_samples_test, is_train=False)
    for i in range(n_users):
        acc_test = cal_acc(dnn_model.predict(ae_list[i].transform(batch_xs)), batch_ys)
        print(i, acc_test)
