import json
import os
import datetime
import threading

import migrant_script.parsers as parsers
import yaml
from urllib.parse import urlparse
from trains import Task
import re
from dateutil.tz import tzutc
from PIL import Image


class Migrant(object):
    artifacts = "artifacts"
    metrics = "metrics"
    params = "params"
    tags = "tags"
    general_information = "general_information"
    skip_tags = ["project.backend", "project.entryPoint", "parentRunId", "best_run"]

    def __init__(self, branch, paths):
        self.paths = paths
        self.size = len(paths)
        self.thread_id = 0
        self.branch = branch
        self.__tag_parsers = {
            "source.name": parsers.source_name_parser(self),
            "log-model.history": parsers.log_model_history_tag_parser(self),
        }
        self.info = {}

    def read(self):
        self.thread_id = threading.current_thread().ident
        for id, path in self.paths:
            self.info[id] = {}
            self.__read_general_information(
                id,
                path,
            )
            self.__read_artifacts(id, path + self.artifacts)
            self.__read_metrics(id, path + self.metrics)
            self.__read_params(id, path + self.params)
            self.__read_tags(id, path + self.tags)

    def seed(self):
        for id in self.__get_ids():
            Task.create(project_name="mlflow_migrant", task_name=id)
            self.__transmit_information(id)
            self.__transmit_metrics(id)
            self.__transmit_artifacts(id)

    def __transmit_metrics(self, id):
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        logger = task.get_logger()
        metrics = self.__get_metrics(id)
        for graph_name, series_name, table in metrics:
            for p in table:
                logger.report_scalar(
                    "graph " + graph_name,
                    series_name,
                    iteration=p[0],
                    value=float(p[1]),
                )
        task.completed()

    def __transmit_artifacts(self, id):
        artifacts = self.info[id][self.artifacts]
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        for type, l in artifacts.items():
            if type == "folder":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            # elif type == 'csv':
            #     for name, obj in l:
            #         task.get_logger().report_table(name, "local csv" ,iteration=100,csv=obj)
            elif type == "dataframe":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "image":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)
            elif type == "dictionary":
                for name, obj in l:
                    task.upload_artifact(name=name, artifact_object=obj)

    def __transmit_information(self, id):
        parameters = self.__get_params(id)
        general_information = self.__get_general_information(id)
        artifact = self.__get_artifact(id)
        tags = self.__get_tags(id)
        task = Task.get_task(project_name="mlflow_migrant", task_name=id)
        task_values = task.export_task()
        task_values["comment"] = (
            tags["note.content"] if "note.content" in tags.keys() else ""
        )
        task_values["hyperparams"]["Args"] = parameters
        task_values["started"] = general_information["started"]
        task_values["completed"] = general_information["completed"]
        task_values["script"]["branch"] = (
            tags["source.git.branch"]
            if "source.git.branch" in tags.keys()
            else self.branch
        )
        task_values["script"]["repository"] = (
            tags["source.git.repoURL"] if "source.git.repoURL" in tags.keys() else ""
        )
        task_values["script"]["entry_point"] = tags["entry_point"]
        task_values["script"]["working_dir"] = tags["working_dir"]
        task_values["script"]["requirements"][tags["project.env"]] = (
            artifact["requirements"] if "requirements" in artifact.keys() else ""
        )
        task_values["user"] = tags["user"]

        task.update_task(task_values)

    def __read_general_information(self, id, path):
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            if name.endswith(".yaml"):
                with open(path + os.sep + name) as file:
                    documents = yaml.full_load(file)
                    timestamp_start_time = (
                        int(documents["start_time"]) / 1000
                        if documents["start_time"]
                        else None
                    )
                    timestamp_end_time = (
                        int(documents["end_time"]) / 1000
                        if documents["end_time"]
                        else None
                    )
                    data_time_start = (
                        datetime.datetime.fromtimestamp(
                            timestamp_start_time, tz=tzutc()
                        )
                        if timestamp_start_time
                        else None
                    )
                    data_time_end = (
                        datetime.datetime.fromtimestamp(timestamp_end_time, tz=tzutc())
                        if timestamp_end_time
                        else None
                    )
                    self.info[id][self.general_information] = {
                        "started": data_time_start,
                        "completed": data_time_end,
                    }
                    break

    def __read_artifacts(self, id, path):
        self.info[id][self.artifacts] = {}
        dirs = list(os.walk(path))[0][1]  # returns all the dirs in 'path'
        for dir in dirs:
            if "model" in dir:
                files = list(os.walk(path + os.sep + dir))[0][
                    2
                ]  # returns all the files in 'path'
                for name in files:
                    if name.endswith(".yaml"):
                        with open(path + os.sep + dir + os.sep + name) as file:
                            documents = yaml.full_load(file)
                            self.info[id][self.artifacts]["requirements"] = str(
                                documents
                            )
                            break
                self.__insert_artifact_by_type(id, "folder", dir, path + os.sep + dir)
        files = list(os.walk(path))[0][2]  # returns all the files in 'path'
        for file_name in files:
            if file_name.endswith(".json"):
                with open(path + os.sep + file_name) as json_file:
                    data = json.load(json_file)
                    self.__insert_artifact_by_type(id, "dictionary", file_name, data)
            elif (
                file_name.endswith(".png")
                or file_name.endswith(".jpg")
                or file_name.endswith(".jpeg")
            ):
                im = Image.open(path + os.sep + file_name)
                self.__insert_artifact_by_type(id, "image", file_name, im)

    def __read_metrics(self, id, path):
        self.info[id][self.metrics] = []
        self.__read_all_sub_directories_content(
            lambda id, name, path, path_tree: self.__read_metric_content(
                id, name, path, path_tree
            ),
            path,
            [self.metrics],
            id,
        )

    def __read_metric_content(self, id, name, path, path_tree):
        if name.startswith("."):
            return
        tag = name.strip().replace("mlflow.", "")
        with open(path + os.sep + name) as file:
            lines = file.readlines()
            lines = [x.strip() for x in lines]
            parsers.epochs_summary_parser(self, id, path_tree, tag, lines)

    def __read_all_sub_directories_content(self, reader, path, path_tree, id):
        contents = list(os.walk(path))
        files = contents[0][2]  # returns all the files in 'path'
        for name in files:
            reader(id, name, path, path_tree)
        dirs = contents[0][1]  # returns all the dirs in 'path'
        current_tree = path_tree.copy()
        for dir in dirs:
            current_tree.append(dir)
            self.__read_all_sub_directories_content(
                reader, path + os.sep + dir, current_tree, id
            )
            current_tree.pop()

    def __read_params(self, id, path):
        self.info[id][self.params] = {}
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            tag = name.strip().replace("mlflow.", "")
            with open(path + os.sep + name) as file:
                value = file.readline().strip()
                if re.match(r"^[Ff]ile://", value):
                    p = urlparse(value)
                    value = os.path.abspath(os.path.join(p.netloc, p.path))
                    value = parsers.get_value_from_path(value)
                    if value:
                        self.__insert_artifact(id, value)
                elif re.match(r"^[Hh]ttps?://", value):
                    value = parsers.get_value_from_path(value)
                    if value:
                        self.__insert_artifact(id, value)
                else:
                    # "epochs": {
                    #     "section": "Args",
                    #     "name": "epochs",
                    #     "value": "6",
                    #     "type": "int",
                    #     "description": "number of epochs to train (default: 6)"
                    # }
                    if re.match(r"^\d*\.\d+", value):
                        value = {
                            "section": "Args",
                            "name": tag,
                            "value": value,
                            "type": "float",
                            "description": parsers.get_description(tag, value),
                        }
                    elif re.match(r"^\d+", value):
                        value = {
                            "section": "Args",
                            "name": tag,
                            "value": value,
                            "type": "int",
                            "description": parsers.get_description(tag, value),
                        }
                    elif value == "True" or value == "False":
                        value = {
                            "section": "Args",
                            "name": tag,
                            "value": value,
                            "type": "boolean",
                            "description": parsers.get_description(tag, value),
                        }
                    else:
                        value = {
                            "section": "Args",
                            "name": tag,
                            "value": value,
                            "type": "string",
                            "description": parsers.get_description(tag, value),
                        }
                    self.info[id][self.params][tag] = value
                file.close()

    def __insert_artifact_by_type(self, id, type, name, value):
        if type in self.info[id][self.artifacts].keys():
            self.info[id][self.artifacts][type].append((name, value))
        else:
            self.info[id][self.artifacts][type] = [(name, value)]

    def __insert_artifact(self, id, value):
        type, name, table = value
        self.__insert_artifact_by_type(id, type, name, table)

    def __read_tags(self, id, path):
        self.info[id][self.tags] = {}
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            with open(path + os.sep + name) as file:
                tag = name.strip().replace("mlflow.", "")
                if tag in self.skip_tags:
                    continue
                value = file.readline().strip()
                self.__tag_parsers[tag](
                    id, value
                ) if tag in self.__tag_parsers.keys() else parsers.tag_parser(
                    self, id, tag, value
                )

    def __get_ids(self):
        return self.info.keys()

    def __get_params(self, id):
        return self.info[id][self.params]

    def __get_metrics(self, id):
        return self.info[id][self.metrics]

    def __get_artifact(self, id):
        return self.info[id][self.artifacts]

    def __get_tags(self, id):
        return self.info[id][self.tags]

    def __get_general_information(self, id):
        return self.info[id][self.general_information]
