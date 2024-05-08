
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
import tqdm as tqdm

import loadAndProcessMRI
import tqdmProgressBar

"""
Build a 3D convolutional neural network model.

This function creates a 3D CNN model with multiple convolutional, pooling, and dense layers. The model takes a 3D input of shape (height, width, depth, 1) and outputs a single sigmoid value.

Args:
    width (int): The width of the input images.
    height (int): The height of the input images.
    depth (int): The depth (number of slices) of the input images.

Returns:
    keras.Model: The compiled 3D CNN model.
"""

def get_model(width=256, height=256, depth=64):
    """Build a 3D convolutional neural network model."""
    input_shape = (height, width, depth, 1)  # Include the channel dimension
    inputs = keras.Input(shape=input_shape)  # Pass the input shape without the batch size

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

"""
Trains an image classifier using 3D convolutional neural networks (3D CNNs) on MRI scan data.

Args:
    path (str): The path to the directory containing the MRI scan data.

Returns:
    None
"""

def train_image_classifier(path):
    hc_path = [os.path.join(path, "HC", x) for x in os.listdir((path + "/HC"))]
    print("HC path found")
    pd_path = [os.path.join(path, "PD", x) for x in os.listdir((path + "/PD"))]
    print("PD path found")

    pd_scans = np.array([loadAndProcessMRI.process_scan(path) for path in pd_path])
    hc_scans = np.array([loadAndProcessMRI.process_scan(path) for path in hc_path])
    print("Scans placed into numpy arrays")

    pd_labels = np.array([1 for _ in range(len(pd_scans))])
    hc_labels = np.array([0 for _ in range(len(hc_scans))])
    print("Labels generated")

    x_train = np.concatenate((pd_scans[:24], hc_scans[:24]), axis=0)
    y_train = np.concatenate((pd_labels[:24], hc_labels[:24]), axis=0)
    x_val = np.concatenate((pd_scans[6:], hc_scans[6:]), axis=0)
    y_val = np.concatenate((pd_labels[6:], hc_labels[6:]), axis=0)
    # Convert NumPy arrays to TensorFlow tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

    batch_size = 2
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_tensor)).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_tensor, y_val_tensor)).batch(batch_size)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    batch_size = 2
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(loadAndProcessMRI.train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(loadAndProcessMRI.validation_preprocessing)
    .batch(batch_size, drop_remainder=True)
    .prefetch(2)
    )

    
    data = train_dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    print("Dimension of the MRI scan is:", image.shape)
    #plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
    #plot_slices(4, 10, 256, 256, image[:, :, :40])

    # Build model.
    print("building model")
    model = get_model(width=256, height=256, depth=64)
    model.summary()

    print("Shape of training dataset:")
    for data in train_dataset:
        print(data[0].shape, data[1].shape)

    print("Shape of validation dataset:")
    for data in validation_dataset:
        print(data[0].shape, data[1].shape)


    print("Compiling Model")
    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )
    print("defining call backs")
    # Define callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    tqdm_progress_bar = tqdmProgressBar.myTQDMProgressBar()

    epochs=5

    # Convert NumPy arrays to TensorFlow tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_train_tensor = tf.expand_dims(x_train_tensor, axis=0)
    x_train_tensor = tf.expand_dims(x_train_tensor, axis=-1)

    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_train_tensor = tf.expand_dims(y_train_tensor, axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_tensor))

    x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)

    x_val_tensor = tf.expand_dims(x_val_tensor, axis=0)
    x_val_tensor = tf.expand_dims(x_val_tensor, axis=-1)

    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)
    y_val_tensor = tf.expand_dims(y_val_tensor, axis=0)

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_tensor, y_val_tensor))

    # Train the model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=0,  # Set verbose to 0 to remove default progress bar
        callbacks=[checkpoint_cb, early_stopping_cb, tqdm_progress_bar],
    )
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])