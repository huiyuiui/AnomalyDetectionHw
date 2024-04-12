import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy.fft import fft, ifft

np.random.seed(0)

def resample_plot(plot_data):
    # Resample data to plot
    plot_label = plot_data[:, 0].flatten()
    plot_data = plot_data[:, 1:]
    outlier_ratio = 0.1 if category == "Wafer" else 0.2
    plot_data, plot_label = resample(plot_data, plot_label, outlier_ratio=outlier_ratio, target_label=1)

    return plot_data, plot_label

def KNN(train_data, test_data, k):
    # Calculate distance
    distances_matrix = pairwise_distances(test_data, train_data, n_jobs=-1)
    k_nearest = np.sort(distances_matrix)[:, :k]
    anomaly_score = np.mean(k_nearest, axis=1)
    
    return anomaly_score

def PCA_Reconstruction(train_data, test_data, n):
    pca = PCA(n)
    pca.fit(train_data)
    transform_data = pca.transform(test_data)
    reconstruct_test = pca.inverse_transform(transform_data)
    anomaly_score = []
    for reconstruct, origin in zip(reconstruct_test, test_data):
        anomaly_score.append(distance.euclidean(reconstruct, origin))
    
    return np.mean(anomaly_score), reconstruct_test

def DFT(test_data, m, k):
    frequency_data = fft(test_data)
    lowest_data = frequency_data[:, :m]
    magnitude = np.abs(lowest_data)
    distances_matrix = pairwise_distances(magnitude, magnitude, n_jobs=-1)
    k_nearest = np.sort(distances_matrix)[:, :k]
    anomaly_score = np.mean(k_nearest, axis=1)
    
    # for visualization purpose
    reconstruct_data = np.zeros_like(frequency_data)
    reconstruct_data[:, :m] = lowest_data
    
    return anomaly_score, reconstruct_data
    
def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    Resample the data to balance classes.

    Parameters:
        data: np.array, shape=(n_samples, n_features)
            Input data.
        label: np.array, shape=(n_samples,)
            Labels corresponding to the data samples.
        outlier_ratio: float, optional (default=0.01)
            Ratio of outliers to include in the resampled data.
        target_label: int, optional (default=0)
            The label to be treated as normal.

    Returns:
        new_data: np.array
            Resampled data.
        new_label: np.array
            Resampled labels.
    """
    new_data = []
    new_label = []
    for i in [1, -1]:
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * outlier_ratio
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False
            )
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * 1)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * 0)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

def Visualization(data, label, subtitle, n_samples=10):
    # Get normal and abnormal samples
    normal_data = data[label == 0]
    abnormal_data = data[label == 1]

    # Randomly select samples
    normal_indices = np.random.choice(normal_data.shape[0], n_samples, replace=False)
    abnormal_indices = np.random.choice(abnormal_data.shape[0], n_samples, replace=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(subtitle)

    # Plot abnormal samples
    for idx in abnormal_indices:
        axes[0].plot(abnormal_data[idx], 'r-')
    axes[0].set_title('Anomaly Sample')

    # Plot normal samples
    for idx in normal_indices:
        axes[1].plot(normal_data[idx], 'b-')
    axes[1].set_title('Normal Sample')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # Load the data
    category = "ECG200" # Wafer / ECG200
    # category = "Wafer"
    print(f"Dataset: {category}")
    train_data = pd.read_csv(f'./{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
    test_data = pd.read_csv(f'./{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()
    plot_data, plot_label = resample_plot(test_data)

    train_label = train_data[:, 0].flatten()
    train_data = train_data[:, 1:]
    train_data, train_label = resample(train_data, train_label, outlier_ratio=0.0, target_label=1)

    test_label = test_data[:, 0].flatten()
    test_data = test_data[:, 1:]
    test_data, test_label = resample(test_data, test_label, outlier_ratio=0.1, target_label=1)

    # # KNN
    # anomaly_score = KNN(train_data, test_data, 5)
    # score = roc_auc_score(test_label, anomaly_score)
    # print("KNN with k=5 score:", score)
    
    # PCA
    min_dist = 0
    min_n = 0
    for n in range(10, min(test_data.shape[0], test_data.shape[1]), 10):
        anomaly_score, _ = PCA_Reconstruction(train_data, test_data, n)
        print(f"PCA with n={n} score: {anomaly_score}")
        if(min_dist > anomaly_score): 
            min_dist = anomaly_score
            min_n = n
    print(f"PCA with n={min_n} has min dist: {min_dist}\n")

    
    # DFT
    max_score = 0
    max_n = 0
    for n in range(10, min(test_data.shape[0], test_data.shape[1]), 10):
        anomaly_score, _ = DFT(test_data, n, 5)
        score = roc_auc_score(test_label, anomaly_score)
        print(f"DFT with n={n} and k=5 score: {score}")
        if(max_score < score): 
            max_score = score
            max_n = n
    print(f"DFT with n={max_n} and k=5 has max score: {max_score}\n")
    
    # Raw Data Visualization
    # Visualization(plot_data, plot_label, category, n_samples=10)
    
    # PCA Visualization
    # _, reconstruct_data = PCA_Reconstruction(train_data, plot_data, 50)
    # Visualization(reconstruct_data, plot_label, category, n_samples=10)
    
    # DFT Visualization
    _, reconstruct_data = DFT(plot_data, 20, 5)
    reconstruct_data = ifft(reconstruct_data).real
    Visualization(reconstruct_data, plot_label, category, n_samples=10)