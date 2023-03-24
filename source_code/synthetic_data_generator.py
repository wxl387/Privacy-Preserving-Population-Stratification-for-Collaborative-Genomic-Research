import pandas as pd
import numpy as np

import time



def gen_syn_attr(X, attr, lambda_):
    unique_val = np.unique(X[:, attr], return_counts=True)
    prob = unique_val[1] / sum(unique_val[1])
    shift = int(np.round(len(unique_val[0]) * lambda_))
    prob_s = np.roll(prob, shift)
    # np.random.seed(0)
    new_attr = np.random.choice(unique_val[0].tolist(), size=X.shape[0], p=prob_s)
    return new_attr


def gen_syn_data(X, y, attrs, lambda_):
    training = np.append(X, y, 1)

    # groupby y and split x
    unique_y = np.unique(y)
    X_groupby_y = []
    for val in unique_y:
        X_groupby_y.append(training[training[:, -1] == val])

        # update each group individually with label-based prob distribution
    for group in X_groupby_y:
        for attr in attrs:
            group[:, attr] = gen_syn_attr(group, attr, lambda_)

    # merge new X and shuffle X+y
    train_new = X_groupby_y[0].copy()
    if len(X_groupby_y) > 1:
        for i in range(1, len(X_groupby_y)):
            train_new = np.concatenate((train_new, X_groupby_y[i]), axis=0)

    np.random.shuffle(train_new)
    X_new_shuffle = train_new[:, 0:-1]
    y_shuffle = train_new[:, -1]
    y_shuffle = np.reshape(y_shuffle, (-1, 1))
    return X_new_shuffle, y_shuffle

if __name__ == "__main__":
    race = pd.read_csv('afr_eur_eas_chr22.csv')
    label = np.reshape([[0] * 200, [1] * 200, [2] * 200], (-1, 1))
    race['label'] = label

    X = race.iloc[:, 0: -1].to_numpy()
    y = race.iloc[:, -1].to_numpy()
    y = np.reshape(y, (-1, 1))

    #---------------------------------------#

    # 3000
    start = time.time()

    X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
    synData = np.append(X_new, y_new, 1)
    size_mul = 5
    for i in range(size_mul - 1):
        X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
        data = np.append(X_new, y_new, 1)
        synData = np.append(synData, data, 0)

    end = time.time()
    print("--- %s seconds ---" % (end - start))

    test_3000 = pd.DataFrame(synData)
    test_3000.rename(columns={37751: "label"}, inplace=True)

    # test_3000.to_csv('race_PCA_test_3000.csv', index=False)

    #---------------------------------------#

    # 6000
    start = time.time()

    X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
    synData = np.append(X_new, y_new, 1)
    size_mul = 10
    for i in range(size_mul - 1):
        X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
        data = np.append(X_new, y_new, 1)
        synData = np.append(synData, data, 0)

    end = time.time()
    print("--- %s seconds ---" % (end - start))

    test_6000 = pd.DataFrame(synData)
    test_6000.rename(columns={37751: "label"}, inplace=True)

    # test_6000.to_csv('race_PCA_test_6000.csv', index=False)

    #---------------------------------------#

    # 9000
    start = time.time()

    X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
    synData = np.append(X_new, y_new, 1)
    size_mul = 15
    for i in range(size_mul - 1):
        X_new, y_new = gen_syn_data(X, y, np.arange(0, X.shape[1]), 0)
        data = np.append(X_new, y_new, 1)
        synData = np.append(synData, data, 0)

    end = time.time()
    print("--- %s seconds ---" % (end - start))

    test_9000 = pd.DataFrame(synData)
    test_9000.rename(columns={37751: "label"}, inplace=True)

    # test_9000.to_csv('race_PCA_test_9000.csv', index=False)
