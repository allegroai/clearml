import fire
from clearml import Task


def add(x, y):
    return x + y


def multiply(x, y):
    return x * y


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="fire dict command")
    fire.Fire(
        {
            "add": add,
            "multiply": multiply,
        }
    )
