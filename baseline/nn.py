from uci_adult import ADULT
from models.dnn import DNN, train_dnn


if __name__ == '__main__':
    adult = ADULT(path='../adult', n_users=10, user_id=0)

    dnn_model = DNN(
        input_size=adult.n_features,
        output_size=adult.n_labels,
        architecture={'h1': 128, 'h2': 128},
        learning_rate=0.001
    )
    train_dnn(dnn_model, [adult], batch_size=128, epochs=10000, display_step=10)

