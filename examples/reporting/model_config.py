# ClearML - Example of manual model configuration
#
import os

from clearml import Task, OutputModel


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='Model configuration example')

# Connect a local configuration file
config_file = os.path.join('data_samples', 'sample.json')
config_file = task.connect_configuration(config_file)
# then read configuration as usual, the backend will contain a copy of it.
# later when executing remotely, the returned `config_file` will be a temporary file
# containing a new copy of the configuration retrieved form the backend
# # model_config_dict = json.load(open(config_file, 'rt'))

# Or Store dictionary of definition for a specific network design
model_config_dict = {
    'value': 13.37,
    'dict': {'sub_value': 'string', 'sub_integer': 11},
    'list_of_ints': [1, 2, 3, 4],
}
model_config_dict = task.connect_configuration(model_config_dict)

# We now update the dictionary after connecting it, and the changes will be tracked as well.
model_config_dict['new value'] = 10
model_config_dict['value'] *= model_config_dict['new value']

# store the label enumeration of the training model
labels = {'background': 0, 'cat': 1, 'dog': 2}
task.connect_label_enumeration(labels)

# Manually log a local model file
OutputModel().update_weights('my_best_model.bin')

# storing a model: Any saved model (keras / pytorch / tensorflow / etc.)
# will have the task network configuration and label enumeration
print('Any model stored from this point onwards, will contain both model_config and label_enumeration')
