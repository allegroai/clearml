# ClearML - Example of manual model reporting
from clearml import Task, OutputModel

# Connecting ClearML with the current process,
task = Task.init(project_name="examples", task_name="Model reporting example")

# Create output model and connect it to the task
output_model = OutputModel(task=task)

labels = {"background": 0, "cat": 1, "dog": 2}
output_model.update_labels(labels)

model_url = "https://allegro-examples.s3.amazonaws.com/clearml-public-resources/v1.0/clearml-examples-open/newexamples/examples/pytorch%20lightning%20mnist%20example.fb969db720e241e5859d522aa5226b81/models/training.pt"

# Manually log a model file, which will have the labels connected above
output_model.update_weights(register_uri=model_url)
