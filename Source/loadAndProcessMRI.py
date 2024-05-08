
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt

"""
Rotate a 3D volume using random angles.

Args:
    volume (tf.Tensor): A 3D tensor representing the volume to be rotated.

Returns:
    tf.Tensor: The rotated volume.
"""

@tf.function
def rotate(volume):
    def scipy_rotate(volume):
            # define some rotation angles
            angles = [-20, -10, -5, 5, 10, 20]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            return volume.astype(np.float32)  # Cast to float32


    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

"""
Plots a set of MRI slices in a grid layout.

Args:
    num_rows (int): The number of rows in the grid.
    num_columns (int): The number of columns in the grid.
    width (int): The width of each slice.
    height (int): The height of each slice.
    data (numpy.ndarray): The 4D array of MRI slice data.

Returns:
    None
"""
def plot_slices(num_rows, num_columns, width, height, data):
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

"""Process validation data by only adding a channel."""
def train_preprocessing(volume, label):
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

"""
Preprocess the input volume and label for validation.

Args:
    volume (tf.Tensor): The input volume to preprocess.
    label (tf.Tensor): The label corresponding to the input volume.

Returns:
    Tuple[tf.Tensor, tf.Tensor]: The preprocessed volume and label.
"""
def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


"""
Read a NIfTI file and return the raw data.

Args:
    filepath (str): The path to the NIfTI file.

Returns:
    numpy.ndarray: The raw data from the NIfTI file.
"""
def read_nii_file(filepath):
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

"""
Normalize the input volume by scaling the values between 0 and 1.

Args:
    volume (numpy.ndarray): The input volume to be normalized.

Returns:
    numpy.ndarray: The normalized volume.
"""
def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

"""
Resizes a 3D MRI volume to a desired depth, width, and height.

Args:
    img (numpy.ndarray): The 3D MRI volume to be resized.

Returns:
    numpy.ndarray: The resized 3D MRI volume.
"""
def resize_volume(img):
    # Set the desired depth
    desired_depth = 64
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nii_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume