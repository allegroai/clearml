import json
import yaml
from clearml import Task, OutputModel


# Connecting ClearML with the current process,
task = Task.init(project_name='examples', task_name='Model configuration example')


# Connect a local configuration file in json format
config_file_json = 'data_samples/sample.json'

# In the web UI, this file will appear in the CONFIGURATION OBJECTS tab,
# under the "json file" subsection because of the `name` parameter entered here
task.connect_configuration(name="json file", configuration=config_file_json)

# Read configuration as usual, the backend will contain a copy of it.
# When executing remotely, the returned `config_file_json` will be a temporary file
# that contains a new copy of the configuration retrieved form the backend
model_config_dictionary_json = json.load(open(config_file_json, 'rt'))

# Create an output model for the PyTorch framework
output_model = OutputModel(task=task, framework="PyTorch", config_dict=model_config_dictionary_json)


# Connecting a local configuration file in yaml format
config_file_yaml = 'data_samples/config_yaml.yaml'
# task.connect_configuration(configuration=config_file_yaml, name="yaml file")

# Read configuration as usual
model_config_dictionary_yaml = yaml.load(open(config_file_yaml), Loader=yaml.FullLoader)

# Connecting a dictionary of definitions for a specific network design
model_config_dict = {
    'CHANGE ME': 13.37,
    'dict': {'sub_value': 'string', 'sub_integer': 11},
    'list_of_ints': [1, 2, 3, 4],
}
model_config_dict = task.connect_configuration(name='dictionary', configuration=model_config_dict)

# Update the dictionary after connecting it, and the changes will be tracked as well.
model_config_dict['new value'] = 10
model_config_dict['CHANGE ME'] *= model_config_dict['new value']

# Connecting label enumeration
labels = {'background': 0, 'cat': 1, 'dog': 2}
output_model.update_labels(labels=labels)

# Manually log a local model file, which will have the labels connected above
OutputModel().update_weights('my_best_model.bin')

# Any saved model (keras / pytorch / tensorflow / etc.) will have the task network configuration and label enumeration
print('Any model stored from this point onwards, will contain both model_config and label_enumeration')