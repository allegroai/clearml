# ClearML - Example of Python Fire integration, processing commands derived from a class
#
from clearml import Task

import fire


class BrokenCalculator(object):
    def __init__(self, offset=1):
        self._offset = offset

    def add(self, x, y):
        return x + y + self._offset

    def multiply(self, x, y):
        return x * y + self._offset


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire class command")
    fire.Fire(BrokenCalculator)
