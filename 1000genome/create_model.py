import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import time
import joblib


# split data into different groups based on labels
def df_split(df, label):
    return df.loc[(df['label'] == label)].reset_index(drop=True).copy()

# train the model
def combine_model(df, n_pc):
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    pca_model = PCA(n_components=n_pc).fit(X)
    return pca_model

# tranform the test data points
def transform(df, pca):
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    principalComponents = np.insert(pca.transform(X), n_pc, y, axis=1)

    return principalComponents

if __name__ == "__main__":
    # read in data
    df_mix = pd.read_csv('afr_eur_eas_chr22.csv', index_col=0)

    # normalization
    scaler = StandardScaler()
    scaler.fit(df_mix)
    df_mix_X_std = scaler.transform(df_mix)

    # create ground truth
    # reduce dimension by PCA
    n_pc = 10
    pca = PCA(n_components=n_pc)
    principalComponents = pca.fit_transform(df_mix_X_std)
    # find labels by kmeans
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(principalComponents)

    # re-assign labels
    df_assign = df_mix.copy()
    df_assign['label'] = y_kmeans

    df_red = df_split(df_assign, 0)
    df_green = df_split(df_assign, 1)
    df_blue = df_split(df_assign, 2)

    # Split each group into half
    df_red_1 = df_red.sample(n=50)
    df_red_2 = df_red.loc[~df_red.index.isin(df_red_1.index)]

    df_green_1 = df_green.sample(n=50)
    df_green_2 = df_green.loc[~df_green.index.isin(df_green_1.index)]

    df_blue_1 = df_blue.sample(n=50)
    df_blue_2 = df_blue.loc[~df_blue.index.isin(df_blue_1.index)]

    # Save the model as a pickle in a file
    start = time.time()
    PCA_train = pd.concat([df_red_1, df_green_1, df_blue_1])
    trained_model = combine_model(PCA_train, n_pc)
    # joblib.dump(trained_model, f'race_model_pc{n_pc}.pkl')
    end = time.time()
    print("--- %s seconds ---" % (end - start))

    # Save train and test data
    PCA_train_150 = pd.concat([df_red_1, df_green_1, df_blue_1]).reset_index(drop=True)
    # PCA_train_150.to_csv('race_PCA_train_150.csv', index=False)

    PCA_test_450 = pd.concat([df_red_2, df_green_2, df_blue_2]).reset_index(drop=True)
    # PCA_test_450.to_csv('race_PCA_test_450.csv', index=False)

    PCA_test_150 = PCA_test_450.groupby('label').sample(n=50).reset_index(drop=True)
    # PCA_test_150.to_csv('race_PCA_test_150.csv', index=False)

    PCA_test_300 = PCA_test_450.groupby('label').sample(n=100).reset_index(drop=True)
    # PCA_test_300.to_csv('race_PCA_test_300.csv', index=False)
