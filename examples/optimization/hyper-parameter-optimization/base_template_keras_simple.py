# ClearML - Keras with Tensorboard example code, automatic logging model and Tensorboard outputs
#
# Train a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs
# (there is *a lot* of margin for parameter tuning).
# 2 seconds per epoch on a K520 GPU.
from __future__ import print_function

import tempfile
import os

import tensorflow as tf  # noqa: F401
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from clearml import Task, Logger


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='Keras HP optimization base')


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

args = {'batch_size': 128,
        'epochs': 6,
        'layer_1': 512,
        'layer_2': 512,
        'layer_3': 10,
        'layer_4': 512,
        }
args = task.connect(args)

model = Sequential()
model.add(Dense(args['layer_1'], input_shape=(784,)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(args['layer_2']))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(args['layer_3']))
model.add(Activation('softmax'))

model2 = Sequential()
model2.add(Dense(args['layer_4'], input_shape=(784,)))
model2.add(Activation('relu'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Advanced: setting model class enumeration
labels = dict(('digit_%d' % i, i) for i in range(10))
task.set_model_label_enumeration(labels)

output_folder = os.path.join(tempfile.gettempdir(), 'keras_example')

board = TensorBoard(log_dir=output_folder, write_images=False)
model_store = ModelCheckpoint(filepath=os.path.join(output_folder, 'weight.hdf5'))

history = model.fit(X_train, Y_train,
                    batch_size=args['batch_size'], epochs=args['epochs'],
                    callbacks=[board, model_store],
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
Logger.current_logger().report_scalar(title='evaluate', series='score', value=score[0], iteration=args['epochs'])
Logger.current_logger().report_scalar(title='evaluate', series='accuracy', value=score[1], iteration=args['epochs'])
