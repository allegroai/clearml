# TRAINS - Keras with Tensorboard example code, automatic logging model and Tensorboard outputs
#
# Train a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs
# (there is *a lot* of margin for parameter tuning).
# 2 seconds per epoch on a K520 GPU.
from __future__ import print_function

import numpy as np
import tensorflow

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model, save_model, model_from_json

from trains import Task


class TensorBoardImage(TensorBoard):
    @staticmethod
    def make_image(tensor):
        import tensorflow as tf
        from PIL import Image
        tensor = np.stack((tensor, tensor, tensor), axis=2)
        height, width, channels = tensor.shape
        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channels,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs={}):
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)
        import tensorflow as tf
        images = self.validation_data[0]  # 0 - data; 1 - labels
        img = (255 * images[0].reshape(28, 28)).astype('uint8')

        image = self.make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag='image', image=image)])
        self.writer.add_summary(summary, epoch)


batch_size = 128
nb_classes = 10
nb_epoch = 6

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
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

# Connecting TRAINS
task = Task.init(project_name='examples', task_name='Keras with TensorBoard example')
# setting model outputs
labels = dict(('digit_%d' % i, i) for i in range(10))
task.set_model_label_enumeration(labels)

board = TensorBoard(histogram_freq=1, log_dir='/tmp/histogram_example', write_images=False)
model_store = ModelCheckpoint(filepath='/tmp/histogram_example/weight.{epoch}.hdf5')

# load previous model, if it is there
try:
    model.load_weights('/tmp/histogram_example/weight.1.hdf5')
except:
    pass

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    callbacks=[board, model_store],
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
