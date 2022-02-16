# ClearML - Example of Python Fire integration, with commands grouped inside classes
#
from clearml import Task

import fire


class Other(object):
    def status(self):
        return "Other"


class IngestionStage(object):
    def __init__(self):
        self.other = Other()

    def run(self):
        return "Ingesting! Nom nom nom..."

    def hello(self, hello_str):
        return hello_str


class DigestionStage(object):
    def run(self, volume=1):
        return " ".join(["Burp!"] * volume)

    def status(self):
        return "Satiated."


class Pipeline(object):
    def __init__(self):
        self.ingestion = IngestionStage()
        self.digestion = DigestionStage()

    def run(self):
        self.ingestion.run()
        self.digestion.run()


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire grouping command")
    fire.Fire(Pipeline)
