"""Keras Tuner CIFAR10 example for the TensorFlow blog post."""

import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds
from clearml.external.kerastuner import TrainsTunerLogger

from clearml import Task

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(hp):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs
    for i in range(hp.Int('conv_blocks', 3, 5, default=3)):
        filters = hp.Int('filters_' + str(i), 32, 256, step=32)
        for _ in range(2):
            x = tf.keras.layers.Convolution2D(
              filters, kernel_size=(3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(
      hp.Int('hidden_size', 30, 100, step=10, default=50),
      activation='relu')(x)
    x = tf.keras.layers.Dropout(
      hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(
        hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
    return model


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init('examples', 'kerastuner cifar10 tuning')

tuner = kt.Hyperband(
    build_model,
    project_name='kt examples',
    logger=TrainsTunerLogger(),
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=6)

data = tfds.load('cifar10')
train_ds, test_ds = data['train'], data['test']


def standardize_record(record):
    return tf.cast(record['image'], tf.float32) / 255., record['label']


train_ds = train_ds.map(standardize_record).cache().batch(64).shuffle(10000)
test_ds = test_ds.map(standardize_record).cache().batch(64)

tuner.search(train_ds,
             validation_data=test_ds,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1),
                        tf.keras.callbacks.TensorBoard(),
                        ])

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
