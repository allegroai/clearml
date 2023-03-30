# ClearML - example code for logging into "CONFIGURATION":
# - ArgumentParser parameter logging
# - user properties logging
# - logging of hyperparameters via dictionary
# - logging of hyperparameters via TaskParameters
# - logging of configuration objects via TaskParameters
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from argparse import ArgumentParser
from enum import Enum


from clearml import Task
from clearml.task_parameters import TaskParameters, param, percent_param


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='FirstTrial', task_name='first_trial')


# -----------------------------------------------
#  Report user properties
# -----------------------------------------------

task.set_user_properties(custom1='great', custom2=True)
task.set_user_properties(custom3=1, custom4=2.0)


# -----------------------------------------------
#  Report hyperparameters via dictionary
# -----------------------------------------------

class StringEnumClass(Enum):
    A = 'a'
    B = 'b'


class IntEnumClass(Enum):
    C = 1
    D = 2

parameters = {
    'list': [1, 2, 3],
    'dict': {'a': 1, 'b': 2},
    'tuple': (1, 2, 3),
    'int': 3,
    'float': 2.2,
    'string': 'my string',
    'IntEnumParam': StringEnumClass.A,
    'StringEnumParam': IntEnumClass.C
}
parameters = task.connect(parameters)

# adding new parameter after connect (will be logged as well)
parameters['new_param'] = 'this is new'

# changing the value of a parameter (new value will be stored instead of previous one)
parameters['float'] = '9.9'
print(parameters)


# -----------------------------------------------
#  Report hyperparameters via TaskParameters
# -----------------------------------------------

# Define a class that inherits from TaskParameters.
# Note that TaskParameters inherits from _AttrsMeta;
# because of `iterations` and `target_accuracy` do not need to be explicitly populated in an __init__ method.
# Consult the documentation at https://www.attrs.org for more information.
class MyTaskParameters(TaskParameters):

    iterations = param(
        type=int,
        desc="Number of iterations to run",
        range=(0, 100000),
    )

    target_accuracy = percent_param(
        desc="The target accuracy of the model",
    )


my_task_parameters = MyTaskParameters(iterations=1000, target_accuracy=0.95)
my_task_parameters = task.connect(my_task_parameters, name='from TaskParameters-like object')


# -----------------------------------------------
#  Report configuration objects via dictionary
# -----------------------------------------------

complex_nested_dict_configuration = {
    'list_of_dicts': [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}, {'e': 5, 'f': 6}],
    'nested_dicts': {'nested': {'key': 'value', 'extra': 'value'}, 'number': 42},
    'dict': {'simple': 'value', 'number': 2},
    'list': [1, 2, 3],
    'int': 3,
    'float': 2.2,
    'string': 'additional string',
}
complex_nested_dict_configuration = task.connect_configuration(
    complex_nested_dict_configuration, name='configuration dictionary')
print(complex_nested_dict_configuration)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--argparser_int_value', help='integer value', type=int, default=1)
    parser.add_argument('--argparser_disabled', action='store_true', default=False, help='disables something')
    parser.add_argument('--argparser_str_value', help='string value', default='a string')

    args = parser.parse_args()

    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info), file=sys.stderr)
    task_params = task.get_parameters()
    print("Task parameters are: {}".format(task_params))
