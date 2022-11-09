# ClearML - Example of manual model reporting
from clearml import Task, OutputModel

# Connecting ClearML with the current process,
task = Task.init(project_name="examples", task_name="Model reporting example")

# Create output model and connect it to the task
output_model = OutputModel(task=task)

# Optional: add labels to the model, so we do not forget
labels = {"background": 0, "cat": 1, "dog": 2}
output_model.update_labels(labels)

# Register an already existing Model file somewhere
model_url = "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt"
output_model.update_weights(register_uri=model_url)

# Or upload a local model file to be later used
# output_model.update_weights(weights_filename="/path/to/file.onnx")

print("Model registration completed")
