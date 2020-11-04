import os
from abc import ABC, abstractmethod
import threading
import parsers
from trains import Task
import shutil


class Migrant(ABC):
    artifacts = "artifacts"
    metrics = "metrics"
    params = "params"
    tags = "tags"
    general_information = "general_information"
    skip_tags = ["project.backend", "project.entryPoint", "parentRunId", "best_run","model_summary"]

    def __init__(self, paths):
        self.paths = paths
        self.size = len(paths)
        self.thread_id = 0
        self.msgs = {'ERROR':[], 'FAILED':[], 'SUCCESS':[]}
        self.project_link = None
        self.tag_parsers = {
            "source.name": parsers.source_name_parser(self),
            "log-model.history": parsers.log_model_history_tag_parser(self),
        }
        self.info = {}
        self.mlflow_url = None
        self.branch = None
        super().__init__()

    @abstractmethod
    def read(self):
        self.thread_id = threading.current_thread().ident

        for id, path in self.paths:
            task = Task.get_task(project_name="mlflow_migrant", task_name=id)
            if task:
                self.msgs['FAILED'].append('task '+ id +' already exist, if you want to migrate it again, you can archive it in Allegro Trains')
                continue
            self.info[id] = {}
            self.read_general_information(id, path)
            self.read_artifacts(id, path + self.artifacts)
            self.read_metrics(id, path + self.metrics)
            self.read_params(id, path + self.params)
            self.read_tags(id, path + self.tags)

    @abstractmethod
    def seed(self):
        for id in self.get_ids():
            task = Task.create(project_name="mlflow_migrant", task_name=id)
            self.transmit_information(id)
            self.transmit_metrics(id)
            self.transmit_artifacts(id)
            task.mark_started()
            task.completed()
            output_log_web_page = task.get_output_log_web_page()
            self.msgs['SUCCESS'].append(output_log_web_page)
            url_parts = output_log_web_page.split('projects')
            project_id = url_parts[1].split('/')[1]
            self.project_link = url_parts[0] + '/projects/' + project_id


    @abstractmethod
    def transmit_metrics(self, id):
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        logger = task.get_logger()
        metrics = self.get_metrics(id)
        for graph_name, series_name, table in metrics:
            for p in table:
                logger.report_scalar(
                    "graph " + graph_name,
                    series_name,
                    iteration=p[0],
                    value=float(p[1]),
                )
        task.completed()

    @abstractmethod
    def transmit_artifacts(self, id):
        artifacts = (
            self.info[id][self.artifacts]
            if self.artifacts in self.info[id].keys()
            else {}
        )
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        for type, l in artifacts.items():
            if type == "folder":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == 'text':
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "dataframe":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "image":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "dictionary":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "storage-server":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)

    @abstractmethod
    def transmit_information(self, id):
        parameters = self.get_params(id)
        # general_information = self.get_general_information(id)
        artifact = self.get_artifact(id)
        tags = self.get_tags(id)
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        task_values = task.export_task()
        task_values["comment"] = (
            tags["note.content"] if "note.content" in tags.keys() else ""
        )
        task_values["hyperparams"]["Args"] = parameters
        # task_values["started"] = general_information["started"]
        # task_values["completed"] = general_information["completed"]
        task_values["script"]["branch"] = (
            tags["source.git.branch"]
            if "source.git.branch" in tags.keys()
            else self.branch
        )
        task_values["script"]["repository"] = (
            tags["source.git.repoURL"] if "source.git.repoURL" in tags.keys() else ""
        )
        task_values["script"]["version_num"] = (
            tags["source.git.commit"] if "source.git.commit" in tags.keys() else ""
        )
        task_values["script"]["entry_point"] = tags["entry_point"]
        task_values["script"]["working_dir"] = tags["working_dir"]
        if "project.env" in tags.keys():
            task_values["script"]["requirements"][tags["project.env"]] = (
                artifact["requirements"] if "requirements" in artifact.keys() else ""
            )
        task_values["user"] = tags["user"]
        task.update_task(task_values)

    @abstractmethod
    def read_general_information(self, id, path):
        pass

    @abstractmethod
    def read_artifacts(self, id, path):
        pass

    @abstractmethod
    def read_metrics(self, id, path):
        pass

    @abstractmethod
    def read_params(self, id, path):
        pass

    @abstractmethod
    def read_tags(self, id, path):
        pass

    @abstractmethod
    def insert_artifact_by_type(self, id, type, name, value):
        if type in self.info[id][self.artifacts].keys():
            self.info[id][self.artifacts][type].append((name, value))
        else:
            self.info[id][self.artifacts][type] = [(name, value)]

    @abstractmethod
    def insert_artifact(self, id, value):
        type, name, table = value
        self.insert_artifact_by_type(id, type, name, table)

    @abstractmethod
    def get_ids(self):
        return self.info.keys()

    @abstractmethod
    def get_params(self, id):
        return self.info[id][self.params] if self.params in self.info[id].keys() else {}

    @abstractmethod
    def get_metrics(self, id):
        return (
            self.info[id][self.metrics] if self.metrics in self.info[id].keys() else {}
        )

    @abstractmethod
    def get_artifact(self, id):
        return (
            self.info[id][self.artifacts]
            if self.artifacts in self.info[id].keys()
            else {}
        )

    @abstractmethod
    def get_tags(self, id):
        return self.info[id][self.tags] if self.tags in self.info[id].keys() else {}

    @abstractmethod
    def get_general_information(self, id):
        return (
            self.info[id][self.general_information]
            if self.general_information in self.info[id].keys()
            else {}
        )
