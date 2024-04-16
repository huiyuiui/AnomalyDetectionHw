import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from numpy.fft import fft, ifft, fftfreq

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
    # PCA
    pca = PCA(n)
    pca.fit(train_data)
    transform_data = pca.transform(test_data)
    reconstruct_test = pca.inverse_transform(transform_data) # for visualization purpose

    # Calculate anomaly score
    anomaly_score = []
    for i in range(test_data.shape[0]):
        anomaly_score.append(np.linalg.norm(test_data[i] - reconstruct_test[i]))
    
    return np.array(anomaly_score), reconstruct_test

def DFT(data, m):
    # DFT
    fft_data = fft(data)

    # apply FFT frequency on feature dimension
    frequency = fftfreq(data.shape[1])

    # get lowest m coefficient
    lowest_index = np.argsort(np.abs(frequency))[:m]
    lowest_data = fft_data[:, lowest_index]
    magnitude = np.real(lowest_data)

    # for visualization purpose
    reconstruct_data = np.zeros_like(fft_data)
    reconstruct_data[:, lowest_index] = lowest_data
    
    return magnitude, reconstruct_data

def DFT_Function(train_data, test_data, m, k):
    # get lowest m DFT coefficient
    magnitude_train, _ = DFT(train_data, m)
    magnitude_test, reconstruct_data = DFT(test_data, m)

    # apply KNN anomaly detection
    anomaly_score = KNN(magnitude_train, magnitude_test, k)
    
    return anomaly_score, reconstruct_data

def DWT(data, s, max_pow):
    dwt_coefs = []
    # build DWT table
    for i in range(data.shape[0]):
        approx = data[i, :].tolist() # previous level approximate
        coef = []
        detail_level = [] 
        for j in range(max_pow): # max_pow indicate max level to do
            new_approx = []
            new_detail = []
            for k in range(0, len(approx), 2):
                new_approx.append((approx[k] + approx[k+1]) / 2)
                new_detail.append((approx[k+1] - approx[k]) / 2)
            approx = new_approx
            detail_level.append(new_detail)
        # get S coefficients according to level
        coef.append(approx[0]) # first coef always be the last level aprox
        level_s = int(math.log2(s))
        details_index = max_pow - 1
        for j in range(level_s):
            for detail in detail_level[details_index]:
                coef.append(detail) 
            details_index -= 1
        dwt_coefs.append(coef)

    return np.array(dwt_coefs)


def DWT_Function(train_data, test_data, s, k):
    # compute the next power of 2
    feature_dim = train_data.shape[1]
    max_power = math.ceil(math.log2(feature_dim))
    max_length = int(math.pow(2, max_power))

    # pad train and test data
    padding_width = max_length - feature_dim
    train_data_padded = np.pad(train_data, ((0, 0), (0, padding_width)), 'constant')
    test_data_padded = np.pad(test_data, ((0, 0), (0, padding_width)), 'constant')

    # get dwt_coefficient
    train_dwt = DWT(train_data_padded, s, max_power)
    test_dwt = DWT(test_data_padded, s, max_power)

    # apply KNN anomaly detection
    anomaly_score = KNN(train_dwt, test_dwt, k)

    return anomaly_score


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
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
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

    # KNN
    max_score = 0
    max_k = 0
    for k in range(1, 8):
        anomaly_score = KNN(train_data, test_data, k)
        score = roc_auc_score(test_label, anomaly_score)
        # print(f"KNN with k={k} on {category} score: {score}")
        if(max_score < score):
            max_score = score
            max_k = k
    print(f"KNN with k={max_k} has max score on {category}: {max_score}\n")
    
    # PCA
    max_score = 0
    max_n = 0
    for n in range(5, min(test_data.shape[0], test_data.shape[1]), 5):
        anomaly_score, _ = PCA_Reconstruction(train_data, test_data, n)
        score = roc_auc_score(test_label, anomaly_score)
        # print(f"PCA with n={n} on {category} score: {score}")
        if(max_score < score): 
            max_score = score
            max_n = n
    print(f"PCA with n={max_n} has max score on {category}: {max_score}\n")

    
    # DFT
    max_score = 0
    max_m = 0
    for m in range(1, 40):
        anomaly_score, _ = DFT_Function(train_data, test_data, m=m, k=5)
        score = roc_auc_score(test_label, anomaly_score)
        # print(f"DFT with m={m} and k=5 on {category} score: {score}")
        if(max_score < score): 
            max_score = score
            max_m = m
    print(f"DFT with m={max_m} and k=5 has max score on {category}: {max_score}\n")

    # DWT
    max_score = 0
    max_s = 0
    s = 1
    while(1):
        anomaly_score = DWT_Function(train_data, test_data, s=s, k=4)
        score = roc_auc_score(test_label, anomaly_score)
        # print(f"DWT with s={s} and k=5 on {category} score: {score}")
        if(max_score < score):
            max_score = score
            max_s = s
        s *= 2
        if(s > int(math.pow(2, math.ceil(math.log2(train_data.shape[1]))))):
            break
    print(f"DWT with s={max_s} and k=4 has max score on {category}: {max_score}\n")
    
    # Raw Data Visualization
    Visualization(plot_data, plot_label, category + " Dataset", n_samples=10)
    
    # PCA Visualization
    _, reconstruct_data = PCA_Reconstruction(train_data, plot_data, max_n)
    Visualization(reconstruct_data, plot_label, category + f" PCA={max_n}", n_samples=10)
    
    # DFT Visualization
    _, reconstruct_data = DFT_Function(train_data, plot_data, max_m, 5)
    reconstruct_data = ifft(reconstruct_data).real
    Visualization(reconstruct_data, plot_label, category + f" DFT={max_m}", n_samples=10)