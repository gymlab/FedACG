import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis, entropy, norm, wasserstein_distance
import torch
import torch.nn as nn

def load_and_normalize_data(file_path):
    """
    Loads weight difference data from an .npy file and normalizes it per kernel.
    
    Args:
        file_path (str): Path to the .npy file containing weight difference data.
    
    Returns:
        normalized_data (list): A list of normalized data for each kernel.
    """
    weight_diff_data = np.load(file_path)  # Load saved .npy file
    normalized_data = []

    # Assuming shape [out_channels, in_channels, kernel_h, kernel_w]
    for kernel_data in weight_diff_data:
        kernel_flat = kernel_data.flatten()
        mean = np.mean(kernel_flat)
        std = np.std(kernel_flat)
        if std > 0:  # Avoid division by zero
            normalized_kernel = (kernel_flat - mean) / std
            normalized_data.append(normalized_kernel)
        else:
            normalized_data.append(kernel_flat)  # No normalization if std is 0
    
    return normalized_data

def calculate_tail_index(data):
    """
    Calculates the tail index for a dataset based on extreme values.
    
    Args:
        data (array-like): Flattened kernel data.
    
    Returns:
        tail_index (float): Estimated tail index.
    """
    sorted_data = np.sort(np.abs(data))
    top_10_percent = sorted_data[int(0.9 * len(sorted_data)):]  # Top 10% extreme values
    if len(top_10_percent) > 1:
        tail_index = np.mean(np.log(top_10_percent / top_10_percent[0]))
    else:
        tail_index = 0.0
    return tail_index

class WSQConv2d(nn.Module):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit5 = [0.2455, 0.4734, 0.5989, 0.9206, 0.9904]
    bit6 = [0.2219, 0.3354, 0.4478, 0.8548, 0.8936, 0.9315]
    bit8 = [0.0498, 0.0991, 0.203, 0.3355, 0.5280, 0.9925, 1.3935, 1.4585]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(WSQConv2d, self).__init__()
        
        self.alpha = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.clip_prob = clip_prob
        
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(len(self.alpha))])
        if len(self.alpha) == 1:
            b_combinations = b_combinations.unsqueeze(-1)
        q_values = torch.sum(b_combinations * self.alpha, dim=1)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        
    def clip_by_prob(self, x):

        original_shape = x.shape
        x_flat = x.view(x.size(0), -1)
        abs_x_flat = x_flat.abs()
        topk = max(1, int(self.clip_prob * abs_x_flat.size(1)))
        thresholds = torch.topk(abs_x_flat, topk, dim=1, largest=True, sorted=True).values[:, -1].view(-1, 1)

        # Clip values in parallel
        clipped_x_flat = torch.where(
            x_flat > thresholds, thresholds,
            torch.where(x_flat < -thresholds, -thresholds, x_flat)
        )

        # Reshape back to the original shape
        clipped_x = clipped_x_flat.view(original_shape)

        return clipped_x
    
    def forward(self, x):
        with torch.no_grad():
            if self.clip_prob > 0:
                x = self.clip_by_prob(x)
            x_mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            x = x - x_mean
            raw_x_std = x.view(x.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
            x_std = raw_x_std + 1e-12
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = quantized_x * x_std + x_mean
            
            # masking
            norm_mask = raw_x_std.squeeze(-1).squeeze(-1).squeeze(-1) < 1e-12
            dequantized_x[norm_mask] = 0           
            
        return dequantized_x

def quantify_distribution_difference(kernel_data):
    """
    Quantifies the difference between the normalized kernel data and the standard normal distribution.

    Args:
        kernel_data (array-like): Normalized kernel data.

    Returns:
        kl_divergence (float): Kullback-Leibler divergence between the histograms.
        wasserstein_dist (float): Wasserstein distance between the distributions.
    """
    # Compute histogram for kernel data
    hist_data, bins = np.histogram(kernel_data, bins=50, range=(-4, 4), density=True)
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

def plot_normalized_histograms(normalized_data, layer_name, output_dir="normalized_histograms", method='ws'):
    """
    Plots and saves histograms for normalized data for each kernel, including skewness and standard normal distribution.
    Additionally calculates and prints distribution differences.

    Args:
        normalized_data (list): List of normalized data for each kernel.
        layer_name (str): Name of the layer.
        output_dir (str): Directory to save the histogram images.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_skewness = []
    all_kurtosis = []
    all_mean_median_diff = []
    all_tail_index = []
    all_entropy = []
    all_kl_divergence = []
    all_wasserstein_dist = []
    bits = 3
    
    paq_list = []
    for kernel_idx, kernel_data in enumerate(normalized_data):
        if method == 'wsq':
            quant = WSQConv2d(n_bits=bits, clip_prob=-1)
            kernel_data = torch.from_numpy(kernel_data).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            kernel_data = quant(kernel_data).view(-1).cpu().numpy()

        # Normalize
        mean = np.mean(kernel_data)
        std = np.std(kernel_data)
        if std > 0:  # Avoid division by zero
            kernel_data = (kernel_data - mean) / std
                
        kernel_skewness = skew(kernel_data)  # Calculate skewness
        kernel_kurtosis = kurtosis(kernel_data)  # Calculate kurtosis
        kernel_mean = np.mean(kernel_data)
        kernel_median = np.median(kernel_data)
        kernel_mean_median_diff = np.abs(kernel_mean - kernel_median)
        kernel_tail_index = calculate_tail_index(kernel_data)  # Calculate tail index
        kernel_entropy = entropy(np.histogram(kernel_data, bins=50, density=True)[0])  # Calculate entropy
        kl_divergence, wasserstein_dist = quantify_distribution_difference(kernel_data)  # Distribution differences

        all_skewness.append(abs(kernel_skewness))  # Store absolute skewness
        all_kurtosis.append(kernel_kurtosis)
        all_mean_median_diff.append(kernel_mean_median_diff)
        all_tail_index.append(kernel_tail_index)
        all_entropy.append(kernel_entropy)
        all_kl_divergence.append(kl_divergence)
        all_wasserstein_dist.append(wasserstein_dist)

        if kernel_idx % 16 == 0:
            plt.figure()
            plt.hist(kernel_data, bins=50, alpha=0.75, density=True, label="Kernel Data")
            
            # Plot standard normal distribution
            x = np.linspace(-48, 48, 1000)
            plt.plot(x, norm.pdf(x), label="Standard Normal", linestyle='--', color='red')
            
            plt.title(f"Kernel {kernel_idx} | Skewness: {kernel_skewness:.4f}, Kurtosis: {kernel_kurtosis:.4f}\nMean-Median: {kernel_mean_median_diff:.4f}, Tail Index: {kernel_tail_index:.4f}, Entropy: {kernel_entropy:.4f}\nKL Divergence: {kl_divergence:.4f}, Wasserstein: {wasserstein_dist:.4f}")
            plt.xlabel("Normalized Weight Difference")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the histogram image
            save_path = os.path.join(output_dir, f"{layer_name}_{method}{bits}_kernel_{kernel_idx}.png")
            plt.savefig(save_path)
            plt.close()

    # Print the average metrics for all kernels
    avg_skewness = np.mean(all_skewness)
    avg_kurtosis = np.mean(all_kurtosis)
    avg_mean_median_diff = np.mean(all_mean_median_diff)
    avg_tail_index = np.mean(all_tail_index)
    avg_entropy = np.mean(all_entropy)
    avg_kl_divergence = np.mean(all_kl_divergence)
    avg_wasserstein_dist = np.mean(all_wasserstein_dist)
    print(f"Average Absolute Skewness for {layer_name}: {avg_skewness:.4f}")
    print(f"Average Kurtosis for {layer_name}: {avg_kurtosis:.4f}")
    print(f"Average Mean-Median Difference for {layer_name}: {avg_mean_median_diff:.4f}")
    print(f"Average Tail Index for {layer_name}: {avg_tail_index:.4f}")
    print(f"Average Entropy for {layer_name}: {avg_entropy:.4f}")
    print(f"Average KL Divergence for {layer_name}: {avg_kl_divergence:.4f}")
    print(f"Average Wasserstein Distance for {layer_name}: {avg_wasserstein_dist:.4f}")

def process_specific_file(file_path, output_dir="normalized_histograms", method="ws"):
    """
    Processes a specific .npy file, normalizes the data, and generates histograms.
    
    Args:
        file_path (str): Path to the .npy file.
        output_dir (str): Directory to save the generated histograms.
    """
    layer_name = os.path.basename(file_path).replace("_weight_diff.npy", "")
    normalized_data = load_and_normalize_data(file_path)
    plot_normalized_histograms(normalized_data, layer_name, output_dir=output_dir, method=method)

if __name__ == '__main__':
    specific_file_path = "./tmp/diff_ws_5_100_0.05.npy"  # Replace with your file path
    output_histogram_directory = "./tmp"
    method = "wsq"
    process_specific_file(specific_file_path, output_dir=output_histogram_directory, method=method)
