# ClearML - Example of Python Fire integration, with commands derived from an object
#
from clearml import Task

import fire


class Calculator(object):
    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire object command")
    calculator = Calculator()
    fire.Fire(calculator)
