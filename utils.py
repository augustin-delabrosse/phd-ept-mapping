import numpy as np
import os
import random
import rasterio
import tensorflow as tf
import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt

def overlay_masks(img, mask, bands=[5, 3, 1]):
    # Ensure img is a NumPy array
    img = np.array(img)

    if len(img.shape) == 2:
        # Grayscale image
        rgb = np.stack([img, img, img], axis=-1)
    else:
        num_bands = img.shape[0]  # Assuming the shape is (bands, height, width)
        if num_bands == 1:
            # Grayscale image
            gray = img[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif num_bands == 3:
            # RGB image
            rgb = np.transpose(img, (1, 2, 0))  # Assuming shape is (3, height, width)
        elif num_bands > 3:
            # Use specified bands
            r = img[bands[0]]
            g = img[bands[1]]
            b = img[bands[2]]
            rgb = np.stack([r, g, b], axis=-1)
        else:
            raise ValueError("Unsupported number of bands. Expected 1, 3, or more.")

    # Normalize the RGB image
    rgb = np.clip(rgb, 0, 255)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
    rgb = rgb.astype(np.uint8)
    
    # Convert to PyTorch tensors
    rgb_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    mask_tensor = torch.from_numpy(mask.astype(bool))
    
    # Overlay the mask on the image
    img_with_mask = draw_segmentation_masks(rgb_tensor, masks=mask_tensor, alpha=0.4, colors=[(255, 0, 0)])
    
    # Convert back to NumPy array
    overlaid_mask = np.transpose(np.array(img_with_mask), (1, 2, 0))
    
    return overlaid_mask


def inspect_dataset(dataset, dataset_name, num_samples=1, for_autoencoder=False):
    print(f"Inspecting {dataset_name}...")

    if for_autoencoder:
        for i, image in enumerate(dataset.take(num_samples)):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {image.shape}")
            print(f"Image data type: {image.dtype}")
            print(f"Image max value: {tf.reduce_max(image).numpy()}")
            print(f"Image min value: {tf.reduce_min(image).numpy()}")

    else:
        for i, (image, label) in enumerate(dataset.take(num_samples)):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {image.shape}")
            print(f"Label: {label.numpy()}")
            print(f"Image data type: {image.dtype}")
            print(f"Image max value: {tf.reduce_max(image).numpy()}")
            print(f"Image min value: {tf.reduce_min(image).numpy()}")        
    
    print(f"\nFinished inspecting {dataset_name}.")

def read_image(image_path):
    """
    Open and read a raster image file using rasterio.
    
    Parameters:
    - image_path: str, Path to the raster image file.
    
    Returns:
    - rasterio.io.DatasetReader: Rasterio object for accessing image data.
    """
    return rasterio.open(image_path)
    

def _save_patch(patch, profile, index, base_name, output_dir, multispectrale=True):
    """
    Save the extracted patch to a file.
    
    Parameters:
    - patch: np.ndarray, Patch to save.
    - profile: dict, Metadata profile for the patch.
    - index: int, Patch number to use in the filename.
    - base_name: str, Base name for the patch file.
    - multispectrale: bool, If True, treat the patch as multispectral.
    """

    if multispectrale and patch.shape[0] > patch.shape[-1]:
        patch = np.transpose(patch, (2, 0, 1)) 

    # print(patch.shape)
    count = patch.shape[0] if multispectrale else 1
    height = patch.shape[1] if len(patch.shape) > 2 else patch.shape[0]
    width = patch.shape[2] if len(patch.shape) > 2 else patch.shape[1]
    
    assert 1 <= count <= 20
    assert 128 <= height <= 2048
    assert 128 <= width <= 2048
    
    file_path = os.path.join(output_dir, f'{base_name}_{index}.tif')
    with rasterio.open(file_path, 'w', 
                       driver='GTiff', 
                       height=height, 
                       width=width, 
                       count=count, dtype=np.float32, 
                       crs=profile['crs'], 
                       transform=profile['transform']) as dst:
        dst.write(patch, None if multispectrale else 1)


def plot_history(acc_history, val_acc_history, loss_history, val_loss_history):
    # Create a figure with a row of two subplots
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy on the first subplot
    plt.subplot(1, 2, 1)
    plt.plot(acc_history)
    plt.plot(val_acc_history)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    # Plot loss on the second subplot
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('Model Loss')
    plt.ylim((0, 5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def custom_train_test_split(img_paths, dsm_paths, dtm_paths, water_paths, labels, split=0.5):
    """
    Splits image paths, DSM paths, DTM paths, water paths, and labels into training and testing sets.
    
    Args:
        img_paths (list): List of file paths to the image inputs.
        dsm_paths (list): List of file paths to DSM inputs.
        dtm_paths (list): List of file paths to DTM inputs.
        water_paths (list): List of file paths to water data inputs.
        labels (list): List of labels corresponding to each input set.
        split (float): Ratio of data to be used for training. Defaults to 0.5.

    Returns:
        tuple: (X_train, train_labels, X_test, test_labels) where:
            - X_train (tuple): Tuple containing lists of paths for training (image, DSM, DTM, water).
            - train_labels (list): Labels for the training set.
            - X_test (tuple): Tuple containing lists of paths for testing (image, DSM, DTM, water).
            - test_labels (list): Labels for the testing set.
    """

    # Randomly shuffle indices to split the dataset
    selected_indices = random.sample(range(len(img_paths)), len(img_paths))
    train_indices = selected_indices[:int(split * len(img_paths))]
    test_indices = selected_indices[int(split * len(img_paths)):]

    # Prepare training dataset from train indices
    train_image_paths = [img_paths[i] for i in train_indices]
    train_dsm_paths = [dsm_paths[i] for i in train_indices]
    train_dtm_paths = [dtm_paths[i] for i in train_indices]
    train_water_paths = [water_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    # Prepare testing dataset from test indices
    test_image_paths = [img_paths[i] for i in test_indices]
    test_dsm_paths = [dsm_paths[i] for i in test_indices]
    test_dtm_paths = [dtm_paths[i] for i in test_indices]
    test_water_paths = [water_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Combine paths into tuples for easy handling
    X_train = (train_image_paths, train_dsm_paths, train_dtm_paths, train_water_paths)
    X_test = (test_image_paths, test_dsm_paths, test_dtm_paths, test_water_paths)

    return X_train, train_labels, X_test, test_labels


def compute_insect_share(sequence, title):
    labels = []
    for i in range(100000):
        try:
            batch = sequence.__getitem__(i)
            labels += batch[-1].tolist()
        except:
            break 
    print(f"{title}:{round(100*np.sum(labels)/len(labels), 3)}%")