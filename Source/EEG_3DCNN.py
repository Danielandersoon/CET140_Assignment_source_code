import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mne
import matplotlib.pyplot as plt

import tqdmProgressBar

"""
Build a 3D convolutional neural network model for EEG classification.

This function creates a 3D CNN model with multiple convolutional, pooling, and dense layers.
The model takes a 3D input of shape (channels, time_steps, 1) and outputs a single sigmoid value.

Args:
    num_channels (int): The number of channels in the EEG data.
    time_steps (int): The number of time steps in the EEG data.

Returns:
    keras.Model: The compiled 3D CNN model.
"""
def get_model(num_channels=64, time_steps=256):
    input_shape = (num_channels, time_steps, 1)  # Include the channel dimension
    inputs = keras.Input(shape=input_shape)  # Pass the input shape without the batch size

    x = layers.Conv3D(filters=64, kernel_size=(1, 3, 3), activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=(1, 3, 3), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=(1, 3, 3), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="eeg_3dcnn")
    return model

"""
Trains an EEG classifier using 3D convolutional neural networks (3D CNNs) on BDF EEG data.

Args:
    path (str): The path to the directory containing the BDF EEG data.

Returns:
    None
"""
def train_eeg_classifier(path):
    hc_path = [os.path.join(path, "HC", x) for x in os.listdir(os.path.join(path, "HC"))]
    print("HC path found")
    pd_path = [os.path.join(path, "PD", x) for x in os.listdir(os.path.join(path, "PD"))]
    print("PD path found")

    pd_eeg = [process_eeg(path) for path in pd_path]
    hc_eeg = [process_eeg(path) for path in hc_path]
    print("EEG data loaded into memory")

    pd_labels = np.array([1 for _ in range(len(pd_eeg))])
    hc_labels = np.array([0 for _ in range(len(hc_eeg))])
    print("Labels generated")

    #
    #
    # Note to self:
    # Remember! We still gotta define the size of the input data 
    # Should be the same for all of them but double check
    # Remember the struggles you had last time you were here...
    # Remember the nightmare...
    #
    #
    # Pad or truncate the EEG data to a fixed shape
    fixed_shape = (, , )  # Specify the desired fixed shape
    pd_eeg = [np.pad(eeg, [(0, 0), (0, fixed_shape[1] - eeg.shape[1]), (0, 0)], mode='constant') if eeg.shape[1] < fixed_shape[1] else eeg[:, :fixed_shape[1], :] for eeg in pd_eeg]
    hc_eeg = [np.pad(eeg, [(0, 0), (0, fixed_shape[1] - eeg.shape[1]), (0, 0)], mode='constant') if eeg.shape[1] < fixed_shape[1] else eeg[:, :fixed_shape[1], :] for eeg in hc_eeg]
    
    x_train = np.concatenate((pd_eeg[:12], hc_eeg[:12]), axis=0)
    y_train = np.concatenate((pd_labels[:12], hc_labels[:12]), axis=0)
    x_val = np.concatenate((pd_eeg[3:], hc_eeg[4:]), axis=0)
    y_val = np.concatenate((pd_labels[3:], hc_labels[:]), axis=0)

    # Convert NumPy arrays to TensorFlow tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

    batch_size = 2
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_tensor)).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_tensor, y_val_tensor)).batch(batch_size)

    # Build model.
    print("building model")
    model = get_model(num_channels=64, time_steps=256)
    model.summary()

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
        "eeg_classification.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    tqdm_progress_bar = tqdmProgressBar.myTQDMProgressBar()

    epochs = 5

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

"""
Process a BDF EEG file and return the data.

Args:
    filepath (str): The path to the BDF EEG file.

Returns:
    numpy.ndarray: The processed EEG data.
"""
def process_eeg(filepath):
    # Load the BDF file
    if (filepath.endswith(".bdf")):
        print("File is a BDF file")
        raw = mne.io.read_raw_bdf(filepath, preload=True)

        # Resample the data to a desired sampling rate
        raw.resample(256)

        # Apply bandpass filter
        raw.filter(l_freq=1, h_freq=50)

        # Get the data as a NumPy array
        data = raw.get_data()

        # Normalize the data
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

        # Reshape the data to (channels, time_steps, 1)
        data = np.expand_dims(data.T, axis=-1)

        return data
    else:
        print("File is not a BDF file")
        return
    