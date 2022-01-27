import fire
from clearml import Task


class Calculator(object):
    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="fire object command")
    calculator = Calculator()
    fire.Fire(calculator)
