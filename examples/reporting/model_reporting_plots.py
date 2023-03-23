# ClearML - Example of manual model reporting
from clearml import Task, OutputModel
from clearml.utilities.plotly_reporter import SeriesInfo
import pandas as pd
import numpy as np

# Connecting ClearML with the current process,
task = Task.init(project_name="examples", task_name="Model reporting plots example")

# Create output model and connect it to the task
output_model = OutputModel(task=task)

# Optional: add labels to the model, so we do not forget
labels = {"background": 0, "cat": 1, "dog": 2}
output_model.update_labels(labels)

# Register an already existing Model file somewhere
model_url = "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt"
output_model.update_weights(register_uri=model_url)

output_model.report_scalar("Reported Metrics", "Val - mAP@50", 72.7, 0)
output_model.report_single_value("Total epochs", 8)

df = pd.DataFrame(
    {
        "Model": ["YOLOv5n", "YOLOv5s", "YOLOv5m6", "YOLOv5x6"],
        "size (pixels)": [640, 640, 1280, 1280],
        "Val - mAP@50-95": [28.0, 37.4, 51.3, 55.0],
        "CPU Speed b1 (ms)": [45, 98, 887, 3136],
    }
)
output_model.report_table(
    title="Summary Table", series="Comparison", iteration=0, table_plot=df
)

output_model.report_line_plot(
    title="Accuracy",
    series=[
        SeriesInfo(
            name="Validation",
            data=np.array(
                [
                    [0, 0.3],
                    [1, 0.55],
                    [2, 0.7],
                    [3, 0.77],
                    [4, 0.8],
                    [5, 0.816],
                    [6, 0.822],
                    [7, 0.829],
                ]
            ),
        )
    ],
    xaxis="Iteration",
    yaxis="Validation Accuracy",
)

# Or upload a local model file to be later used
# output_model.update_weights(weights_filename="/path/to/file.onnx")

print("Model registration completed")
