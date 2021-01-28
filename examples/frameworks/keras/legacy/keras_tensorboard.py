# ClearML - Keras with Tensorboard example code, automatic logging model and Tensorboard outputs
#
# Train a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs
# (there is *a lot* of margin for parameter tuning).
# 2 seconds per epoch on a K520 GPU.
from __future__ import print_function

import argparse
import tempfile
import os
import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import tensorflow as tf
from clearml import Task


class TensorBoardImage(TensorBoard):
    @staticmethod
    def make_image(tensor):
        from PIL import Image
        import io
        tensor = np.stack((tensor, tensor, tensor), axis=2)
        height, width, channels = tensor.shape
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channels,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)
        images = self.validation_data[0]  # 0 - data; 1 - labels
        img = (255 * images[0].reshape(28, 28)).astype('uint8')

        image = self.make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag='image', image=image)])
        self.writer.add_summary(summary, epoch)


parser = argparse.ArgumentParser(description='Keras MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train (default: 6)')
args = parser.parse_args()

# the data, shuffled and split between train and test sets
nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32')/255.
X_test = X_test.reshape(10000, 784).astype('float32')/255.
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model2 = Sequential()
model2.add(Dense(512, input_shape=(784,)))
model2.add(Activation('relu'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='Keras with TensorBoard example')
task.connect_configuration({'test': 1337, 'nested': {'key': 'value', 'number': 1}})

# Advanced: setting model class enumeration
labels = dict(('digit_%d' % i, i) for i in range(10))
task.set_model_label_enumeration(labels)

output_folder = os.path.join(tempfile.gettempdir(), 'keras_example')

board = TensorBoard(histogram_freq=1, log_dir=output_folder, write_images=False)
model_store = ModelCheckpoint(filepath=os.path.join(output_folder, 'weight.{epoch}.hdf5'))

# load previous model, if it is there
# noinspection PyBroadException
try:
    model.load_weights(os.path.join(output_folder, 'weight.1.hdf5'))
except Exception:
    pass

history = model.fit(X_train, Y_train,
                    batch_size=args.batch_size, epochs=args.epochs,
                    callbacks=[board, model_store],
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
