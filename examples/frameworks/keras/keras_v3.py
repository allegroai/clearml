import numpy as np
import keras
from clearml import Task


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model

Task.init(project_name="examples", task_name="keras_v3")

model = get_model()

test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

model.save("my_model.keras")

reconstructed_model = keras.models.load_model("my_model.keras")

np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
