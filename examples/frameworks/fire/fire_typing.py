from typing import Tuple, List
from clearml import Task
import fire


def with_ret() -> Tuple:
    print("With ret called")
    return 1, 2


def with_args(arg1: int, arg2: List):
    print("With args called", arg1, arg2)


def with_args_and_ret(arg1: int, arg2: List) -> Tuple:
    print("With args and ret called", arg1, arg2)
    return 1, 2


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire typing command")
    fire.Fire()
