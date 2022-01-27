import fire
from clearml import Task


def hello(count, name="clearml", prefix="prefix_", suffix="_suffix", **kwargs):
    for _ in range(count):
        print("Hello %s%s%s!" % (prefix, name, suffix))


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="fire single command")
    fire.Fire(hello)
