# TRAINS - Example of manual model configuration
#
import os
from tempfile import gettempdir

import torch
from trains import Task


task = Task.init(project_name='examples', task_name='Manual model configuration')

# create a model
model = torch.nn.Module

# store dictionary of definition for a specific network design
model_config_dict = {
    'value': 13.37,
    'dict': {'sub_value': 'string'},
    'list_of_ints': [1, 2, 3, 4],
}
task.set_model_config(config_dict=model_config_dict)

# or read form a config file (this will override the previous configuration dictionary)
# task.set_model_config(config_text='this is just a blob\nof text from a configuration file')

# store the label enumeration the model is training for
task.set_model_label_enumeration({'background': 0, 'cat': 1, 'dog': 2})
print('Any model stored from this point onwards, will contain both model_config and label_enumeration')

# storing the model, it will have the task network configuration and label enumeration

torch.save(model, os.path.join(gettempdir(), "model"))
print('Model saved')
