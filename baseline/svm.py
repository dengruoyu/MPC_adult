import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from time import time

from uci_adult import ADULT


def svm(x_train, y_train):
    print('SVM begin')
    t0 = time()
    model = SVC(kernel="rbf")
    model.fit(x_train, y_train)
    print("training time:", round(time() - t0, 3), "s")
    return model


def cal_acc(model, x_test, y_test):
    # make predictions
    expected = y_test
    predicted = model.predict(x_test)
    # summarize the fit of the model
    print("accuracy score:", metrics.accuracy_score(expected, predicted))
    print("recall score:", metrics.recall_score(expected, predicted))


if __name__ == '__main__':
    adult = ADULT(path='../adult', n_users=100, user_id=0)
    svm_model = svm(adult.x_train, np.argmax(adult.y_train, axis=1))
    cal_acc(svm_model, adult.x_test, np.argmax(adult.y_test, axis=1))
