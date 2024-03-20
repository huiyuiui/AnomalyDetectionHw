from sklearn.metrics import pairwise_distances


def KNN(train_data, test_data, k):
    # Calculate distance
    distances_matrix = pairwise_distances(test_data, train_data, n_jobs=-1)
    k_nearest = np.sort(distances_matrix)[:, :k]
    anomaly_score = np.mean(k_nearest, axis=1)
    sorted_index = np.argsort(anomaly_score)[::-1]
    # Pick top n % as anomaly
    n = 10
    top_n = int(test_data.shape[0] * n * 0.01)
    anomaly_index = sorted_index[:top_n]
    pred = np.zeros(test_data.shape[0], dtype=int)
    pred[anomaly_index] = 1

    return pred


def Kmeans(train_data, test_data, k):
    # Construct clusters
    k_clusters = train_data[np.random.choice(train_data.shape[0], k, replace=False)]
    for i in range(100):
        distance_matrix = pairwise_distances(train_data, k_clusters, n_jobs=-1)
        assigned_cluster = np.argmin(distance_matrix, axis=1)
        k_clusters = np.array(
            [train_data[assigned_cluster == j].mean(axis=0) for j in range(k)]
        )
    # Pick n% distance as threshold
    n = 90
    train_distances = pairwise_distances(train_data, k_clusters, n_jobs=-1)
    train_min_distances = np.min(train_distances, axis=1)
    threshold = np.percentile(train_min_distances, n)
    # Calculate distance
    test_distances = pairwise_distances(test_data, k_clusters, n_jobs=-1)
    min_distances = np.min(test_distances, axis=1)
    pred = (min_distances > threshold).astype(int)

    return pred


def Cosine_distance(test_data, k=5):
    # Calculate cosine distance
    norms = np.linalg.norm(test_data, axis=1, keepdims=True)
    normalized_data = test_data / norms
    cosine_similarity_matrix = np.dot(normalized_data, normalized_data.T)
    cosine_distance_matrix = (1 - cosine_similarity_matrix)  # map value to 0(same) ~ 2(different)
    # Find anomaly
    kth_nearest = np.sort(cosine_distance_matrix)[:, k-1]
    sorted_index = np.argsort(kth_nearest)[::-1]
    # Pick top n % as anomaly
    n = 10
    top_n = int(test_data.shape[0] * n * 0.01)
    pred = np.zeros(test_data.shape[0], dtype=int)
    anomaly_index = sorted_index[:top_n]
    pred[anomaly_index] = 1
    
    return pred    

########################################################
########  Do not modify the sample code segment ########
########################################################

import torchvision
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score

seed = 0
np.random.seed(seed)


def resample_total(data, label, ratio=0.05):
    """
    data: np.array, shape=(n_samples, n_features)
    label: np.array, shape=(n_samples,)
    ratio: float, ratio of samples to be selected
    """
    new_data = []
    new_label = []
    for i in range(10):
        i_data = data[label == i]
        idx = np.random.choice(
            list(range(len(i_data))), int(len(i_data) * ratio), replace=False
        )
        new_data.append(i_data[idx])
        new_label.append(np.ones(len(idx)) * i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    data: np.array, shape=(n_samples, n_features)
    label: np.array, shape=(n_samples,)
    outlier_ratio: float, ratio of outliers
    target_label: int, the label to be treated as normal
    """
    new_data = []
    new_label = []
    for i in range(10):
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * ((outlier_ratio / 9))
            idx = np.random.choice(list(range(len(i_data))), int(num), replace=False)
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * i)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


if __name__ == "__main__":
    orig_train_data = torchvision.datasets.MNIST(
        "MNIST/",
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        target_transform=None,
        download=True,
    )  # 下載並匯入MNIST訓練資料
    orig_test_data = torchvision.datasets.MNIST(
        "MNIST/",
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        target_transform=None,
        download=True,
    )  # 下載並匯入MNIST測試資料

    orig_train_label = orig_train_data.targets.numpy()
    orig_train_data = orig_train_data.data.numpy()
    orig_train_data = orig_train_data.reshape(60000, 28 * 28)

    orig_test_label = orig_test_data.targets.numpy()
    orig_test_data = orig_test_data.data.numpy()
    orig_test_data = orig_test_data.reshape(10000, 28 * 28)

    # PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=30)
    pca_data = pca.fit_transform(np.concatenate([orig_train_data, orig_test_data]))
    orig_train_data = pca_data[: len(orig_train_label)]
    orig_test_data = pca_data[len(orig_train_label) :]

    orig_train_data, orig_train_label = resample_total(
        orig_train_data, orig_train_label, ratio=0.1
    )

    KNN_1_score = []
    KNN_5_score = []
    KNN_10_score = []
    Kmeans_1_score = []
    Kmeans_5_score = []
    Kmeans_10_score = []
    Cosine_distance_score = []
    for i in tqdm.tqdm(range(10)):
        train_data = orig_train_data[orig_train_label == i]
        test_data, test_label = resample(
            orig_test_data, orig_test_label, target_label=i, outlier_ratio=0.1
        )
        # [TODO] prepare training/testing data with label==i labeled as 0, and others labeled as 1
        train_label = np.zeros(len(train_data), dtype=int)
        for j in range(len(test_label)):
            if test_label[j] == i:
                test_label[j] = 0
            else:
                test_label[j] = 1
        test_label = np.array(test_label, dtype=int)

        # [TODO] implement methods
        # [TODO] record ROC-AUC for each method

        # KNN
        # for k in [1, 5, 10]:
        #     pred = KNN(train_data, test_data, k)
        #     score = roc_auc_score(test_label, pred)
        #     if(k == 1): KNN_1_score.append(score)
        #     elif(k == 5): KNN_5_score.append(score)
        #     else: KNN_10_score.append(score)

        # K means
        # for k in [1, 5, 10]:
        #     pred = Kmeans(train_data, test_data, k)
        #     score = roc_auc_score(test_label, pred)
        #     if(k == 1): Kmeans_1_score.append(score)
        #     elif(k == 5): Kmeans_5_score.append(score)
        #     else: Kmeans_10_score.append(score)

        # Distance based
        pred = Cosine_distance(test_data, k=5)
        score = roc_auc_score(test_label, pred)
        Cosine_distance_score.append(score)

    # [TODO] print the average ROC-AUC for each method
    # print("KNN with k=1 score:", np.mean(KNN_1_score))
    # print("KNN with k=5 score:", np.mean(KNN_5_score))
    # print("KNN with k=10 score:", np.mean(KNN_10_score))
    # print("Kmeans with k=1 score:", np.mean(Kmeans_1_score))
    # print("Kmeans with k=5 score:", np.mean(Kmeans_5_score))
    # print("Kmeans with k=10 score:", np.mean(Kmeans_10_score))
    print("Cosine Distance with k=5 score:", np.mean(Cosine_distance_score))
