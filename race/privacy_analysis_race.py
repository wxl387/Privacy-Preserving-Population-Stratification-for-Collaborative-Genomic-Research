import pandas as pd
import numpy as np

import math
import joblib
import time
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
import warnings

def transform(arr, pca):
    X = arr[:, 0:-1]
    y = arr[:, -1]
    principalComponents = np.insert(pca.transform(X), n_pc, y, axis=1)
    return principalComponents

def random_sample(arr, n_sample, replacement):
    idx = np.random.choice(arr.shape[0], size=n_sample, replace=replacement)
    return arr[idx, :]

def data_sampling(population, model, pop_a_label, pop_b_label, pop):
    pop_a = population[population.label == pop_a_label]
    pop_a_1 = pop_a.sample(frac=0.5)
    pop_a_2 = pop_a.loc[~pop_a.index.isin(pop_a_1.index)]

    pop_b = population[population.label == pop_b_label]
    pop_b_1 = pop_b.sample(frac=0.5)
    pop_b_2 = pop_b.loc[~pop_b.index.isin(pop_b_1.index)]

    if pop == 1:
        researcher_1 = pop_a_1
        researcher_2 = pop_b_1
    elif pop == 2:
        researcher_1 = pd.concat([pop_a_1, pop_b_1])
        researcher_2 = pd.concat([pop_a_2, pop_b_2])

    N_researcher = researcher_1.shape[0]
    not_r1 = population[~population.index.isin(researcher_1.index)]
    df_control = not_r1.groupby('label').sample(n=int(N_researcher / 3) - 1)

    r1_transform_start = time.time()
    researcher_1 = transform(researcher_1.to_numpy(), model)
    r1_transform_end = time.time()
    r1_transform_time = r1_transform_end - r1_transform_start

    r2_transform_start = time.time()
    researcher_2 = transform(researcher_2.to_numpy(), model)
    r2_transform_end = time.time()
    r2_transform_time = r2_transform_end - r2_transform_start

    control_group = transform(df_control.to_numpy(), model)
    case_group = random_sample(researcher_1, control_group.shape[0], False)

    return researcher_1, researcher_2, control_group, case_group, r1_transform_time, r2_transform_time

def euclidean_distance(arr1, arr2):
    list_1 = arr1[:, 0:-1]
    list_2 = arr2[:, 0:-1]

    dist_arr = np.array([])

    for i in range(list_1.shape[0]):
        min_dist = 9999999
        point1 = list_1[i]
        for j in range(list_2.shape[0]):
            point2 = list_2[j]
            dist = np.linalg.norm(point1 - point2)
            if min_dist > dist:
                min_dist = dist
        dist_arr = np.append(dist_arr, [min_dist])
    return dist_arr

def power_cal(researcher, control, case):
    quantile_val = 0.05
    dist_control = euclidean_distance(control, researcher)
    threshold = np.quantile(dist_control, quantile_val)
    dist_case = euclidean_distance(case, researcher)
    count = dist_case[dist_case < threshold].shape[0]
    power = count / dist_case.shape[0]
    return power


def laplace_noise(deviation, eps):
    lambda_ = deviation / eps
    noise = np.random.laplace(loc=0, scale=lambda_)
    return noise


def add_noise(arr, eps, sensitivity):
    col_diff = arr.max(axis=0) - arr.min(axis=0)
    new_arr = np.zeros((arr.shape[0], arr.shape[1]))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            noise = laplace_noise(col_diff[j], eps)
            new_arr[i, j] = arr[i, j] + noise

    new_arr[:, -1] = arr[:, -1]

    return new_arr


def util_cal(truth, k_labels):
    # Prep
    k_labels_matched = np.empty_like(k_labels)
    # For each cluster label...
    for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((k_labels == k) * (truth == t)) for t in np.unique(truth)]
        k_labels_matched[k_labels == k] = np.unique(truth)[np.argmax(match_nums)]

    cm = confusion_matrix(truth, k_labels_matched)

    return cm


def exp_run(model, common_ds, clustering, control_group, case_group, researcher_1, researcher_2, pop_a_label,
            pop_b_label, sensitivity):
    # random pick noise from laplace distribution
    exp = 20
    epsilon = np.arange(0.1, 50, 0.5)
    n = len(epsilon)
    power_arr = np.zeros((exp, n))
    precision_arr = np.zeros((exp, n))
    recall_arr = np.zeros((exp, n))

    for i in range(exp):
        for j in range(n):
            eps = epsilon[j]

            r1_add_noise_start = time.time()
            researcher_1_new = add_noise(researcher_1, eps, sensitivity)
            r1_add_noise_end = time.time()
            r1_add_noise_time = r1_add_noise_start - r1_add_noise_end

            r2_add_noise_start = time.time()
            researcher_2_new = add_noise(researcher_2, eps, sensitivity)
            r2_add_noise_end = time.time()
            r2_add_noise_time = r2_add_noise_start - r2_add_noise_end

            server_pred_start = time.time()
            target = np.concatenate((researcher_1_new, researcher_2_new), axis=0)
            r1_label = pop_a_label
            r2_label = pop_b_label

            population_new = np.concatenate((common_ds, target), axis=0)

            X_test = population_new[:, 0:-1]
            y_test = population_new[:, -1]
            y_pred = clustering.predict(X_test)
            server_pred_end = time.time()
            server_pred_time = server_pred_end - server_pred_start

            cf_matrix = util_cal(y_test, y_pred)

            if sum(cf_matrix[:, r1_label]) + sum(cf_matrix[:, r2_label]) != 0:
                precision = (cf_matrix[r1_label, r1_label] + cf_matrix[r2_label, r2_label]) / (
                            sum(cf_matrix[:, r1_label]) + sum(cf_matrix[:, r2_label]))
            else:
                precision = 1.0

            if sum(cf_matrix[r1_label, :]) + sum(cf_matrix[r2_label, :]) != 0:
                recall = (cf_matrix[r1_label, r1_label] + cf_matrix[r2_label, r2_label]) / (
                            sum(cf_matrix[r1_label, :]) + sum(cf_matrix[r2_label, :]))
            else:
                recall = 1.0

            precision_arr[i, j] = precision
            recall_arr[i, j] = recall

            power = power_cal(researcher_1_new, control_group, case_group)
            power_arr[i, j] = power

            df_precision = pd.DataFrame(precision_arr, columns=epsilon)
            df_recall = pd.DataFrame(recall_arr, columns=epsilon)
            df_power = pd.DataFrame(power_arr, columns=epsilon)

            df_precision_mean = pd.DataFrame(df_precision.mean()).reset_index()
            df_precision_mean.columns = ['epsilon', 'value']
            df_precision_mean['metric'] = 'precision'

            df_recall_mean = pd.DataFrame(df_recall.mean()).reset_index()
            df_recall_mean.columns = ['epsilon', 'value']
            df_recall_mean['metric'] = 'recall'

            df_power_mean = pd.DataFrame(df_power.mean()).reset_index()
            df_power_mean.columns = ['epsilon', 'value']
            df_power_mean['metric'] = 'power'
        # end for
        # end for

    return df_precision_mean, df_recall_mean, df_power_mean, r1_add_noise_time, r2_add_noise_time, server_pred_time

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    common_ds = pd.read_csv('race_PCA_train_150.csv')
    for n_test in range(1, 3):
        population = pd.read_csv(f'race_PCA_test_{n_test * 150}.csv')

        for n_pop in range(1, 3):
            n_pc_list = [2]
            for n_pc in n_pc_list:
                model = joblib.load(f'race_model_pc{n_pc}.pkl')
                common_ds_pca = transform(common_ds.to_numpy(), model)
                kmeans_train = common_ds_pca[:, 0:-1]
                population_pca = transform(population.to_numpy(), model)
                col_diff_general = common_ds.max(axis=0) - common_ds.min(axis=0)

                for n_cluster in range(3, 4):
                    kmeans = KMeans(n_clusters=n_cluster)
                    kmeans.fit(kmeans_train)

                    df_precision = pd.DataFrame()
                    df_recall = pd.DataFrame()
                    df_power = pd.DataFrame()

                    print('n_test: ' + str(n_test * 150) + ', n_pop: ' + str(n_pop) + ', n_pc: ' + str(
                        n_pc) + ', n_cluster: ' + str(n_cluster))

                    r1_time_list = 0
                    r2_time_list = 0

                    server_time_list = 0

                    draw_times = 20
                    for draw_time in range(draw_times):
                        # random pick 2 labels
                        pop_a_label, pop_b_label = np.random.choice(3, 2, replace=False)

                        researcher_1, researcher_2, control_group, case_group, r1_transform_time, r2_transform_time = data_sampling(
                            population, model, pop_a_label, pop_b_label, n_pop)

                        precision, recall, power, r1_add_noise_time, r2_add_noise_time, server_pred_time = exp_run(
                            model, common_ds_pca, kmeans, control_group, case_group, researcher_1, researcher_2,
                            pop_a_label, pop_b_label, col_diff_general)

                        df_precision = pd.concat([df_precision, precision])
                        df_recall = pd.concat([df_recall, recall])
                        df_power = pd.concat([df_power, power])

                        r1_time_list += (r1_transform_time + r1_add_noise_time)
                        r2_time_list += (r2_transform_time + r2_add_noise_time)
                        server_time_list += server_pred_time

                    print("---researcher 1 takes %s seconds to transform and add noise---" % (
                                r1_time_list / (draw_times)))
                    print("---researcher 2 takes %s seconds to transform and add noise---" % (
                                r2_time_list / (draw_times)))
                    print(
                        "---server takes %s seconds to combine data and predict---" % (server_pred_time / (draw_times)))

    #                 df_precision = df_precision.drop(columns=['metric'])
    #                 df_recall = df_recall.drop(columns=['metric'])
    #                 df_power = df_power.drop(columns=['metric'])

    #                 df_precision_mean = df_precision.groupby('epsilon').mean().reset_index()
    #                 df_recall_mean = df_recall.groupby('epsilon').mean().reset_index()
    #                 df_power_mean = df_power.groupby('epsilon').mean().reset_index()

    #                 df_precision_mean.to_csv(f'df_precision_mean_test{n_test*90}_pc{n_pc}_pop{n_pop}_nc{n_cluster}.csv')
    #                 df_recall_mean.to_csv(f'df_recall_mean_test{n_test*90}_pc{n_pc}_pop{n_pop}_nc{n_cluster}.csv')
    #                 df_power_mean.to_csv(f'df_power_mean_test{n_test*90}_pc{n_pc}_pop{n_pop}_nc{n_cluster}.csv')

    print('done')
