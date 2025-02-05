import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis, entropy, norm, laplace, wasserstein_distance
import torch
import torch.nn as nn

alpha = '0.3'

def clip_by_prob(x, clip_ratio=0.05):
    """
    Clips the array values by a given ratio of top absolute values.

    Args:
        x (np.ndarray): Input array to clip.
        clip_ratio (float): Ratio of elements to determine the clipping threshold.

    Returns:
        np.ndarray: Clipped array.
    """
    # Flatten the array
    x_flat = x.flatten()
    abs_x_flat = np.abs(x_flat)

    # Calculate the threshold
    topk = max(1, int(clip_ratio * abs_x_flat.shape[0]))
    threshold = np.sort(abs_x_flat)[-topk]

    # Clip values based on the threshold
    clipped_x = np.clip(x, a_min=-threshold, a_max=threshold)

    return clipped_x, threshold

def load_and_normalize_data(file_paths):
    """
    Loads weight difference data from multiple .npy files and normalizes it for the entire layer.
    
    Args:
        file_paths (list): List of paths to the .npy files containing weight difference data.
    
    Returns:
        normalized_data_dict (dict): Dictionary of normalized data for each layer.
    """
    normalized_data_dict = {}

    for file_path in file_paths:
        layer_name = os.path.basename(file_path).replace("_weight_diff.npy", "")
        weight_diff_data = np.load(file_path)  # Load saved .npy file
        layer_data = weight_diff_data.flatten()  # Flatten the entire layer data
        # mean = np.mean(layer_data)
        clipped_data, th = clip_by_prob(layer_data.copy(), 0.0001)
        # layer_data = clipped_data
        mean = np.mean(layer_data)
        # std = np.std(clipped_data)
        std = np.std(layer_data)
        print(f"{file_path}: mean: {mean:.8f}, std: {std:.8f}, edge: {th / std:.4f}")
        maxabs = np.max(np.abs(layer_data))

        # if std > 0:  # Avoid division by zero
        #     normalized_data = (layer_data - mean) / std
        # else:
        #     normalized_data = layer_data  # No normalization if std is 0
        # normalized_data = layer_data / maxabs
        normalized_data = layer_data / std

        normalized_data_dict[layer_name] = normalized_data

    return normalized_data_dict

def quantify_distribution_difference(data):
    """
    Quantifies the difference between the normalized data and the standard normal distribution.

    Args:
        data (array-like): Normalized layer data.

    Returns:
        kl_divergence (float): Kullback-Leibler divergence between the histograms.
        wasserstein_dist (float): Wasserstein distance between the distributions.
    """
    # Compute histogram for layer data
    hist_data, bins = np.histogram(data, bins=200, range=(-6, 6), density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Compute standard normal distribution values at bin centers
    standard_normal_pdf = norm.pdf(bin_centers)

    # Normalize histograms to avoid division by zero in KL divergence
    hist_data += 1e-12
    standard_normal_pdf += 1e-12

    # Compute KL divergence
    kl_divergence = np.sum(hist_data * np.log(hist_data / standard_normal_pdf))

    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance(bin_centers, bin_centers, u_weights=hist_data, v_weights=standard_normal_pdf)

    return kl_divergence, wasserstein_dist

def plot_layer_histograms(normalized_data_dict, output_dir="normalized_histograms"):
    """
    Plots and saves histograms for each normalized layer in a single subplot figure.

    Args:
        normalized_data_dict (dict): Dictionary of normalized data for each layer.
        output_dir (str): Directory to save the combined histogram image.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_layers = len(normalized_data_dict)
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6), sharey=True)

    for idx, (layer_name, data) in enumerate(normalized_data_dict.items()):
        ax = axes[idx] if num_layers > 1 else axes

        # Calculate statistics
        layer_skewness = skew(data)  # Calculate skewness
        layer_kurtosis = kurtosis(data)  # Calculate kurtosis
        kl_divergence, wasserstein_dist = quantify_distribution_difference(data)  # Distribution differences

        # Plot histogram
        ax.hist(
            data, bins=100, alpha=0.7, density=True, label=f"Skew: {layer_skewness:.2f}\nKurt: {layer_kurtosis:.2f}\nKL: {kl_divergence:.2f}\nW: {wasserstein_dist:.2f}", edgecolor='black'
        )
        
        maxabs = np.max(np.abs(data))
        
        # NF4_VALUES = np.array([-1.0000, -0.6962, -0.5257, -0.3946, -0.2849, -0.1892, -0.0931, 0.0000,
        #                 0.0796, 0.1603, 0.2453, 0.3487, 0.4622, 0.5952, 0.7579, 1.0000])
        NF4_VALUES = np.array([-2.6436, -1.9735, -1.5080, -1.1490, -0.8337, -0.5439, -0.2686, 0.,
                               0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9676, 2.6488])
        NF4_VALUES2 = np.array([-1.0000, -0.6962, -0.5257, -0.3946, -0.2849, -0.1892, -0.0931, 0.0000,
                        0.0796, 0.1603, 0.2453, 0.3487, 0.4622, 0.5952, 0.7579, 1.0000]) * maxabs
        
        for nf in NF4_VALUES:
            # 클리핑 threshold 위치에 세로선 추가
            ax.axvline(x=nf, color='green', linestyle='--', linewidth=1.5)
            
        for nf in NF4_VALUES2:
            # 클리핑 threshold 위치에 세로선 추가
            ax.axvline(x=nf, color='orange', linestyle='--', linewidth=1.5)

        # Plot standard normal distribution
        x = np.linspace(-4, 4, 1000)
        ax.plot(x, norm.pdf(x), label="Standard Normal", linestyle='--', color='red')

        ax.set_title(layer_name)
        ax.set_xlabel("Normalized Weight Difference")
        if idx == 0:
            ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()

    # Save the combined histogram image
    save_path = os.path.join(output_dir, f"layer_histograms_{alpha}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Layer histograms saved to {save_path}")


def process_multiple_files(file_paths, output_dir="normalized_histograms"):
    """
    Processes multiple .npy files, normalizes the data, and generates histograms for each layer in a single figure.
    
    Args:
        file_paths (list): List of paths to the .npy files.
        output_dir (str): Directory to save the generated combined histogram.
    """
    normalized_data_dict = load_and_normalize_data(file_paths)
    plot_layer_histograms(normalized_data_dict, output_dir=output_dir)

if __name__ == '__main__':
    file_paths = [
        f"./tmp/diff_conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer1.0.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer2.0.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer3.0.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer4.0.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer1.0.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer2.0.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer3.0.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer4.0.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer1.1.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer2.1.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer3.1.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer4.1.conv2.weight_{alpha}.npy",
        f"./tmp/diff_layer1.1.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer2.1.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer3.1.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer4.1.conv1.weight_{alpha}.npy",
        f"./tmp/diff_layer2.0.downsample.0.weight_0.3.npy",
        f"./tmp/diff_layer3.0.downsample.0.weight_0.3.npy",
        f"./tmp/diff_layer4.0.downsample.0.weight_0.3.npy",
    ]  # Replace with your file paths
    output_histogram_directory = "./tmp"
    process_multiple_files(file_paths, output_dir=output_histogram_directory)
