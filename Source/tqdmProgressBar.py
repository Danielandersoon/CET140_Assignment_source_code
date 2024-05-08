import tqdm
import tensorflow as tf

"""
A custom Keras callback that displays a progress bar using the tqdm library during model training.

This callback is used to display a progress bar that shows the current epoch and the progress within each epoch during model training. It updates the progress bar after each batch and closes it at the end of each epoch.

Args:
    None

Returns:
    None
"""
class myTQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self):
        super(myTQDMProgressBar, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        steps = self.params['steps']
        if steps is None:
            self.progress_bar = tqdm.tqdm(desc="Epoch {}".format(epoch + 1))
        else:
            self.progress_bar = tqdm.tqdm(total=steps, desc="Epoch {}".format(epoch + 1))

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()