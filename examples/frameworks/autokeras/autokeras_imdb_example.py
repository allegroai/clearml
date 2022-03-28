try:
    import autokeras as ak
except ImportError:
    import sys

    raise ImportError(
        "autokeras package is missing, you can install it using pip: pip install autokeras"
        if sys.version_info.minor > 6
        else "AutoKeras does not support python version < 3.7"
    )

import numpy as np
import tensorflow as tf
from tensorflow import keras

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="autokeras", task_name="AutoKeras IMDB example with scalars")


def imdb_raw():
    max_features = 20000
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features, index_from=index_offset
    )
    x_train = x_train
    y_train = y_train.reshape(-1, 1)
    x_test = x_test
    y_test = y_test.reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=str)
    x_test = np.array(x_test, dtype=str)
    return (x_train, y_train), (x_test, y_test)


# Prepare the data.
(x_train, y_train), (x_test, y_test) = imdb_raw()
print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # <START> this film was just brilliant casting <UNK>

# Initialize the TextClassifier
clf = ak.TextClassifier(max_trials=2)

# Tensorboard Callbacks
tb_callbacks = [keras.callbacks.TensorBoard(log_dir="log")]

# Search for the best model.
clf.fit(x_train, y_train, epochs=2, callbacks=tb_callbacks)

# Evaluate on the testing data.
clf_accuracy = clf.evaluate(x_test, y_test)
print("Accuracy:", clf_accuracy)
