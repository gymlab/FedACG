import numpy as np
import matplotlib.pyplot as plt
import os

def load_layer_data(file_path):
    """
    Load weight difference data from a .npy file and return it as a 2D-compatible array.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    data = np.load(file_path)  # Load the weight difference data
    return data

def normalize_values(values):
    """
    Normalizes the values by subtracting the mean and dividing by the standard deviation.
    """
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values  # Avoid division by zero
    return (values - mean) / std

def quantize_values(values, quantization_levels):
    """
    Quantizes the values to the nearest level in the given quantization_levels list.
    """
    quantized_values = np.array([quantization_levels[np.abs(quantization_levels - v).argmin()] for v in values])
    return quantized_values

def plot_layer_2d(file_path, layer_name, quantization_levels, output_dir="./2d_plots"):
    """
    Plots the weight difference data in 2D as a scatter plot for a given layer and saves the figure.
    """
    data = load_layer_data(file_path)
    
    if data is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Ensure data is at least 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    
    values = data.flatten()
    combined_idx = np.arange(len(values))  # Directly use a linear index

    # Normalize values
    values = normalize_values(values)
    
    # Sample only 1000 points randomly
    if len(values) > 500:
        sampled_values = values[:500]
        sampled_combined_idx = combined_idx[:500]
    else:
        sampled_values = values
        sampled_combined_idx = np.arange(len(values))
    
    # Quantize sampled values
    sampled_values = quantize_values(sampled_values, np.array(quantization_levels))
    
    # Determine symmetric x-axis limits
    x_max = 3
    
    # 2D Scatter plot with quantized sampled data
    plt.figure(figsize=(10, 10))
    plt.scatter(sampled_values, sampled_combined_idx, c=sampled_values, cmap='viridis', 
                alpha=1.0, edgecolors='k', linewidths=1., s=150)
    
    # Add vertical line at Parameter Value = 0
    plt.axvline(x=0, color='black', linewidth=2)
    
    # Set x-axis to be symmetric around zero
    plt.xlim(-x_max, x_max)
    
    # 축 숫자 제거
    plt.xticks([])  # x축 숫자 제거
    plt.yticks([])  # y축 숫자 제거
    
    # plt.xlabel("Parameter Value (Normalized & Quantized)")
    # plt.ylabel("Sample Index")
    # plt.title(f"2D Quantized Parameter Distribution - {layer_name}")
    # plt.colorbar(label="Quantized Parameter Value")
    
    # Save the plot
    save_path = os.path.join(output_dir, f"{layer_name}_2d_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"2D plot saved at: {save_path}")

def process_multiple_files(file_paths, quantization_levels, output_dir="./2d_plots"):
    """
    Processes multiple .npy files and generates 2D scatter plots for each layer with quantized values.
    """
    for file_path in file_paths:
        layer_name = os.path.basename(file_path).replace("_weight_diff.npy", "")
        plot_layer_2d(file_path, layer_name, quantization_levels, output_dir=output_dir)

if __name__ == '__main__':
    alpha = '0.3'
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
    
    # Define quantization levels
    quantization_levels = [-2.6536,-1.9735,-1.508,-1.149,-0.8337,-0.5439,-0.2686,0.,
                           0.2303,0.4648,0.7081,0.9663,1.2481,1.5676,1.9676,2.6488]  # Modify this list as needed
    
    output_plot_directory = "./2d_plots"
    process_multiple_files(file_paths, quantization_levels, output_dir=output_plot_directory)
