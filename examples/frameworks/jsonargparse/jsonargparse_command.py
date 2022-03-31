from jsonargparse import CLI
from clearml import Task


class Main:
    def __init__(self, prize: int = 100):
        self.prize = prize

    def person(self, name: str):
        return "{} won {}!".format(name, self.prize)


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="jsonargparse command")
    print(CLI(Main))
