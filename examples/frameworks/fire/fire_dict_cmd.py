# ClearML - Example of Python Fire integration, with commands derived from a dictionary 
#
from clearml import Task

import fire


def add(x, y):
    return x + y


def multiply(x, y):
    return x * y


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire dict command")
    fire.Fire(
        {
            "add": add,
            "multiply": multiply,
        }
    )
