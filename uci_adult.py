import numpy as np


def read_raw_data(path, is_train):
    """
    train: 32561/32562
    test: 16281/16283
    total: 48842/48845
    """
    def continuous2list(item, item_split):
        item_list = [0] * len(item_split)
        if item != '?':
            age = float(item)
            for i in range(len(item_split)):
                if age >= item_split[i] and (i == len(item_split) - 1 or age < item_split[i + 1]):
                    item_list[i] = 1
        return item_list

    def discrete2list(item, class_names):
        item_dict = dict()
        for i, class_name in enumerate(class_names):
            item_dict[class_name] = i
        item_list = [0] * len(item_dict.keys())
        if item in item_dict:
            item_list[item_dict[item]] = 1
        return item_list

    with open(path, 'r') as f:
        lines = f.readlines()

    x = list()
    y = list()
    for i, line in enumerate(lines):
        if is_train:
            items = line[:-1].split(', ')
        else:
            items = line[:-2].split(', ')
        if len(items) == 15:
            x.append(list())
            y.append(list())
            x[-1] += continuous2list(
                items[0],
                item_split=[0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]
            )  # age
            x[-1] += discrete2list(
                items[1],
                class_names=['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                             'Federal-gov', 'Local-gov', 'State-gov',
                             'Without-pay', 'Never-worked']
            )  # workclass
            x[-1] += discrete2list(
                items[3],
                class_names=['Bachelors', 'Some-college', '11th',
                             'HS-grad', 'Prof-school', 'Assoc-acdm',
                             'Assoc-voc', '9th', '7th-8th', '12th',
                             'Masters', '1st-4th', '10th',
                             'Doctorate', '5th-6th', 'Preschool']
            )  # education
            x[-1] += continuous2list(
                items[4],
                item_split=[0, 6, 9, 11]
            )  # education-num
            x[-1] += discrete2list(
                items[5],
                class_names=['Married-civ-spouse', 'Divorced', 'Never-married',
                             'Separated', 'Widowed', 'Married-spouse-absent',
                             'Married-AF-spouse']
            )  # marital-status
            x[-1] += discrete2list(
                items[6],
                class_names=['Tech-support', 'Craft-repair', 'Other-service',
                             'Sales', 'Exec-managerial', 'Prof-specialty',
                             'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                             'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
                             'Protective-serv', 'Armed-Forces']
            )  # occupation
            x[-1] += discrete2list(
                items[7],
                class_names=['Wife', 'Own-child', 'Husband',
                             'Not-in-family', 'Other-relative', 'Unmarried']
            )  # relationship
            x[-1] += discrete2list(
                items[8],
                class_names=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
            )  # race
            x[-1] += discrete2list(
                items[9],
                class_names=['Female', 'Male']
            )  # sex
            x[-1] += continuous2list(
                items[10],
                item_split=[0, 1, 2000, 5000, 10000]
            )  # capital-gain
            x[-1] += continuous2list(
                items[11],
                item_split=[0, 1, 2000]
            )  # capital-loss
            x[-1] += continuous2list(
                items[12],
                item_split=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            )  # hours-per-week
            x[-1] += discrete2list(
                items[13],
                class_names=['United-States', 'Cambodia', 'England',
                             'Puerto-Rico', 'Canada', 'Germany',
                             'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                             'Greece', 'South', 'China',
                             'Cuba', 'Iran', 'Honduras',
                             'Philippines', 'Italy', 'Poland',
                             'Jamaica', 'Vietnam', 'Mexico',
                             'Portugal', 'Ireland', 'France',
                             'Dominican-Republic', 'Laos', 'Ecuador',
                             'Taiwan', 'Haiti', 'Columbia',
                             'Hungary', 'Guatemala', 'Nicaragua',
                             'Scotland', 'Thailand', 'Yugoslavia',
                             'El-Salvador', 'Trinadad&Tobago', 'Peru',
                             'Hong', 'Holand-Netherlands']
            )  # native-country:
            y[-1] += discrete2list(
                items[14],
                class_names=['>50K', '<=50K']
            )  # label
    return np.array(x), np.array(y)


def read_data(path='adult', is_raw=False):
    """
    n_features = 133
    n_labels = 2
    n_samples_train = 32561
    n_samples_test = 16281
    """
    if is_raw:
        x_train, y_train = read_raw_data(path+'/adult.data', is_train=True)
        x_test, y_test = read_raw_data(path+'/adult.test', is_train=False)
        np.save(path+'/x_train.npy', x_train)
        np.save(path+'/y_train.npy', y_train)
        np.save(path+'/x_test.npy', x_test)
        np.save(path+'/y_test.npy', y_test)
    else:
        x_train = np.load(path+'/x_train.npy')
        y_train = np.load(path+'/y_train.npy')
        x_test = np.load(path+'/x_test.npy')
        y_test = np.load(path+'/y_test.npy')
    return x_train, y_train, x_test, y_test


class ADULT:
    def __init__(self, path='adult', n_users=None, user_id=None):
        self.x_train, self.y_train, self.x_test, self.y_test = read_data(path)
        if n_users is not None:
            cnt_samples = self.x_train.shape[0] // n_users
            idx_train = np.arange(user_id*cnt_samples,
                                  min((user_id+1)*cnt_samples, self.x_train.shape[0]))
            self.x_train = self.x_train[idx_train]
            self.y_train = self.y_train[idx_train]
        self.n_samples_train = self.x_train.shape[0]
        self.n_samples_test = self.x_test.shape[0]
        self.n_features = self.x_train.shape[1]
        self.n_labels = self.y_train.shape[1]

    def next_batch(self, batch_size=128, is_train=True):
        if is_train:
            rand_idx = np.random.choice(self.n_samples_train, batch_size, replace=False)
            batch_xs = self.x_train[rand_idx]
            batch_ys = self.y_train[rand_idx]
        else:
            rand_idx = np.random.choice(self.n_samples_test, batch_size, replace=False)
            batch_xs = self.x_test[rand_idx]
            batch_ys = self.y_test[rand_idx]
        return batch_xs, batch_ys


if __name__ == '__main__':
    pass
