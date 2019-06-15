# TRAINS - example code, absl logging
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags
from absl import logging

from trains import Task


FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')
flags.DEFINE_string('another_str', 'My string', 'A string', module_name='test')

task = Task.init(project_name='examples', task_name='absl example')

flags.DEFINE_integer('echo3', 3, 'Text to echo.')
flags.DEFINE_string('echo5', '5', 'Text to echo.', module_name='test')


parameters = {
    'list': [1, 2, 3],
    'dict': {'a': 1, 'b': 2},
    'int': 3,
    'float': 2.2,
    'string': 'my string',
}
parameters = task.connect(parameters)

# adding new parameter after connect (will be logged as well)
parameters['new_param'] = 'this is new'

# changing the value of a parameter (new value will be stored instead of previous one)
parameters['float'] = '9.9'


def main(_):
    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info), file=sys.stderr)
    logging.info('echo is %s.', FLAGS.echo)


if __name__ == '__main__':
    app.run(main)
