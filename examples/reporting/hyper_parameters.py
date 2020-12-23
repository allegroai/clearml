# ClearML - example code, ArgumentParser parameter logging, absl parameter logging, and dictionary parameter logging
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from argparse import ArgumentParser

from absl import app
from absl import flags
from absl import logging

from clearml import Task


FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')
flags.DEFINE_string('another_str', 'My string', 'A string', module_name='test')

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='hyper-parameters example')

flags.DEFINE_integer('echo3', 3, 'Text to echo.')
flags.DEFINE_string('echo5', '5', 'Text to echo.', module_name='test')

parameters = {
    'list': [1, 2, 3],
    'dict': {'a': 1, 'b': 2},
    'tuple': (1, 2, 3),
    'int': 3,
    'float': 2.2,
    'string': 'my string',
}
parameters = task.connect(parameters)

# adding new parameter after connect (will be logged as well)
parameters['new_param'] = 'this is new'

# changing the value of a parameter (new value will be stored instead of previous one)
parameters['float'] = '9.9'
print(parameters)

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


def main(_):
    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info), file=sys.stderr)
    logging.info('echo is %s.', FLAGS.echo)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--argparser_int_value', help='integer value', type=int, default=1)
    parser.add_argument('--argparser_disabled', action='store_true', default=False, help='disables something')
    parser.add_argument('--argparser_str_value', help='string value', default='a string')

    args = parser.parse_args()

    app.run(main)
