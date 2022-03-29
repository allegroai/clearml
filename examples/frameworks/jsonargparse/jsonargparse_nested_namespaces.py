from dataclasses import dataclass
from jsonargparse import ArgumentParser
from clearml import Task


@dataclass
class Arg2:
    opt1: str = "from default 1"
    opt2: str = "from default 2"


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="jsonargparse nested namespaces")
    parser = ArgumentParser()
    parser.add_argument("--arg1.opt1", default="from default 1")
    parser.add_argument("--arg1.opt2", default="from default 2")
    parser.add_argument("--arg2", type=Arg2, default=Arg2())
    parser.add_argument("--not-nested")
    print(parser.parse_args())
