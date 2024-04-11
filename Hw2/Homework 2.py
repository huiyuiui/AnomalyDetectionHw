import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

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

def Visualization(data, dataset_name, n_samples=10):
    # Resample data to plot
    plot_label = data[:, 0].flatten()
    plot_data = data[:, 1:]
    outlier_ratio = 0.1 if dataset_name == "Wafer" else 0.2
    plot_data, plot_label = resample(plot_data, plot_label, outlier_ratio=outlier_ratio, target_label=1)

    # Get normal and abnormal samples
    normal_data = plot_data[plot_label == 0]
    abnormal_data = plot_data[plot_label == 1]

    # Randomly select samples
    normal_indices = np.random.choice(normal_data.shape[0], n_samples, replace=False)
    abnormal_indices = np.random.choice(abnormal_data.shape[0], n_samples, replace=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(dataset_name + " Dataset")

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

    # Visualization
    Visualization(train_data, category, n_samples=10)

    train_label = train_data[:, 0].flatten()
    train_data = train_data[:, 1:]
    train_data, train_label = resample(train_data, train_label, outlier_ratio=0.0, target_label=1)

    test_label = test_data[:, 0].flatten()
    test_data = test_data[:, 1:]
    test_data, test_label = resample(test_data, test_label, outlier_ratio=0.1, target_label=1)

    