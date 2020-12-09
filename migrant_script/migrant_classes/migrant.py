from abc import ABC, abstractmethod
import threading
import parsers
from trains import Task
from trains.model import ARCHIVED_TAG
from tqdm import tqdm

from global_variables import PROJECT_NAME


class Migrant(ABC):
    """
        The ``Migrant`` class is a code template for a Migrant object which,

        The ``Migrant`` class and its methods allow you to create and manage

        .. warning::

        For detailed information about creating Migrant objects, see the following methods:

            -
            -
            -

        .. note::

        """

    artifacts = "artifacts"
    metrics = "metrics"
    params = "params"
    tags = "tags"
    general_information = "general_information"
    skip_tags = ["project.backend", "project.entryPoint", "parentRunId", "best_run","model_summary"]

    def __init__(self, paths, pbar, timer,analysis, project_indicator):
        """
        .. warning::
            **Do not construct Migrant manually!**
            Please use :meth:`MigrantFactory.create`
        """
        self.paths = paths
        self.size = len(paths)
        self.migration_count = 0
        self.thread_id = 0
        self.pbar = pbar
        self.analysis = analysis
        self.project_exist = project_indicator
        self.timer = timer
        self.msgs = {'ERROR':[], 'FAILED':[]}
        self.project_link = None
        self.tag_parsers = {
            "source.name": parsers.source_name_parser(self),
            "log-model.history": parsers.log_model_history_tag_parser(self),
        }
        self.info = {}
        self.ID_to_Name = {}
        self.mlflow_url = None
        self.branch = None
        super().__init__()

    @abstractmethod
    def call_func(self, func_name ,id,func, *args):
        res = None
        if self.analysis:
            self.timer.start(func_name, self.thread_id, id)
            res = func(*args)
            self.timer.end(func_name, self.thread_id, id)
        else:
            res = func(*args)
        return res

    @abstractmethod
    def read(self):
        self.thread_id = threading.current_thread().ident
        for id, path in self.paths:
            self.info[id] = {}

            self.call_func('read_tags', id,
                           lambda id_, path_: self.read_tags(id_, path_),
                           id, path + self.tags)

            if self.project_exist:
                task = self.call_func('Task.get_task', id,
                                      lambda id_: Task.get_task(project_name=PROJECT_NAME, task_name=id_),
                                      self.get_run_name_by_id(id))
                if task:
                    task_tags = task.data.system_tags if hasattr(task.data, 'system_tags') else task.data.tags
                    if not ARCHIVED_TAG in task_tags:
                        del self.info[id]
                        self.msgs['FAILED'].append(
                            'task ' + id + ' already exist, if you want to migrate it again, you can archive it in Allegro Trains')
                        self.pbar.update(1)
                        continue

            self.call_func('read_general_information', id,
                           lambda id_, path_: self.read_general_information(id_, path_),
                           id, path)

            self.call_func('read_artifacts', id,
                           lambda id_, path_: self.read_artifacts(id_, path_),
                           id, path + self.artifacts)

            self.call_func('read_metrics', id,
                           lambda id_, path_: self.read_metrics(id_, path_),
                           id, path + self.metrics)

            self.call_func('read_params', id,
                           lambda id_, path_: self.read_params(id_, path_),
                           id, path + self.params)

    @abstractmethod
    def seed(self):
        for id in self.get_ids():
            if "runName" in self.info[id][self.tags].keys():
                self.ID_to_Name[id] = self.info[id][self.tags]["runName"]

            task = self.call_func('Task.create', id,
                           lambda id_: Task.create(project_name=PROJECT_NAME, task_name=id_),
                           self.get_run_name_by_id(id))

            self.call_func('transmit_information', id,
                           lambda id_: self.transmit_information(id_),
                           id)

            self.call_func('transmit_metrics', id,
                           lambda id_: self.transmit_metrics(id_),
                           id)

            self.call_func('transmit_artifacts', id,
                           lambda id_: self.transmit_artifacts(id_),
                           id)

            task.mark_started()
            task.completed()
            output_log_web_page = task.get_output_log_web_page()
            url_parts = output_log_web_page.split('projects')
            project_id = url_parts[1].split('/')[1]
            self.project_link = url_parts[0] + '/projects/' + project_id
            self.migration_count +=1
            self.pbar.update(1)

    @abstractmethod
    def transmit_metrics(self, id):
        task = self.call_func('Task.get_task', id,
                              lambda id_: Task.get_task(project_name=PROJECT_NAME, task_name=id_),
                              self.get_run_name_by_id(id))
        logger = task.get_logger()
        metrics = self.get_metrics(id)
        for graph_name, series_name, table in metrics:
            for p in table:
                logger.report_scalar(
                    graph_name,
                    series_name,
                    iteration=p[0],
                    value=float(p[1])
                )
        task.completed()

    @abstractmethod
    def transmit_artifacts(self, id):
        artifacts = (
            self.info[id][self.artifacts]
            if self.artifacts in self.info[id].keys()
            else {}
        )
        task = self.call_func('Task.get_task', id,
                              lambda id_: Task.get_task(project_name=PROJECT_NAME, task_name=id_),
                              self.get_run_name_by_id(id))
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
        general_information = self.get_general_information(id)
        artifact = self.get_artifact(id)
        tags = self.get_tags(id)

        task = self.call_func('Task.get_task', id,
                              lambda id_: Task.get_task(project_name=PROJECT_NAME, task_name=id_),
                              self.get_run_name_by_id(id))

        task_values = self.call_func('task.export_task', id,
                              lambda _ : task.export_task(),
                                     self.get_run_name_by_id(id))

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

        self.call_func('task.update_task', id,
                              lambda _task_values : task.update_task(_task_values),
                                     task_values)

        if len(tags["VALUETAG"].keys()) > 0:
            self.call_func('task.connect_configuration', id,
                                lambda _dict: task.connect_configuration(_dict,name="MLflow Tags"),
                                        tags["VALUETAG"])

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

    @abstractmethod
    def get_run_name_by_id(self,id):
        return self.ID_to_Name[id] if id in self.ID_to_Name.keys() else id
