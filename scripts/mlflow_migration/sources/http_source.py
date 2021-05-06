import re

import mlflow
import mlflow.server

from .source import Source
from ..parsers import parse_datetime, epochs_summary_parser, insert_param, tag_parser


class HttpSource(Source):
    def __init__(self, addresses, mlflow_url, pbar, timer, analysis, project_indicator):
        mlflow.tracking.set_tracking_uri(mlflow_url)
        self.runs = {}
        self.__seed(addresses)
        super().__init__(addresses, pbar, timer, analysis, project_indicator)
        self.branch = "Remote_Http_server"

    def __seed(self, addresses):
        for id, _ in addresses:
            mlflow_id = id[1:]
            self.runs[id] = mlflow.tracking.MlflowClient().get_run(mlflow_id)

    def __get_run_by_run_id(self, id):
        return self.runs[id]

    def read_general_information(self, id, _):
        info = self.__get_run_by_run_id(id).info
        start_time = info.start_time
        end_time = info.end_time
        artifact_uri = info.artifact_uri
        data_time_start, data_time_end = parse_datetime(start_time, end_time)

        self.info[id][self.general_information] = {
            "started": data_time_start,
            "completed": data_time_end,
            "artifact_uri": artifact_uri,
            "name": "",  # No supported (MLflow)API for get\set run name.
        }

    def read_artifacts(self, id, _):
        self.info[id][self.artifacts] = {}
        artifact_address = (
            self.info[id][self.general_information]["artifact_uri"]
            if "artifact_uri" in self.info[id][self.general_information].keys()
            else ""
        )
        if re.match(r"^(?:s3:/)|(?:gs:/)|(?:azure:/)", artifact_address):
            artifacts = mlflow.tracking.MlflowClient().list_artifact(id[1:])
            for artifact in artifacts:
                parts = artifact.path.split("/")
                self.insert_artifact_by_type(
                    id, "storage-server", parts[-1], artifact.path
                )

    def read_metrics(self, id, _):
        self.info[id][self.metrics] = []
        run_data = self.__get_run_by_run_id(id).data
        metric_names = run_data.metrics.keys()
        run_id = id[1:]
        for name in metric_names:
            entries = mlflow.tracking.MlflowClient().get_metric_history(run_id, name)
            value_gen = (str(entry.value) for entry in entries)
            parts = (self.metrics + "/" + name).split("/")
            if len(parts) > 3:
                parts = parts[0:-1]
            if len(parts) == 2:
                epochs_summary_parser(
                    self,
                    id,
                    [self.metrics],
                    name,
                    value_gen,
                )
            else:
                epochs_summary_parser(self, id, parts, name, value_gen)

    def read_params(self, id, _):
        self.info[id][self.params] = {}
        params = self.__get_run_by_run_id(id).data.params
        for param_name in params.keys():
            value = params[param_name]
            insert_param(self, id, value, param_name, True)

    def read_tags(self, id, _):
        self.info[id][self.tags] = {"VALUETAG": {}}
        tags = self.__get_run_by_run_id(id).data.tags
        for name in tags.keys():
            if "mlflow." in name:
                tag = name.replace("mlflow.", "")
            elif name.startswith("."):
                continue
            else:
                tag = "VALUETAG_" + name
            self.tag_parsers[tag](
                id, tags[name]
            ) if tag in self.tag_parsers.keys() else tag_parser(
                self, id, tag, tags[name]
            )

    def insert_artifact_by_type(self, id, type, name, value):
        super().insert_artifact_by_type(id, type, name, value)

    def insert_artifact(self, id, value):
        super().insert_artifact(id, value)

    def get_ids(self):
        return super().get_ids()

    def get_params(self, id):
        return super().get_params(id)

    def get_metrics(self, id):
        return super().get_metrics(id)

    def get_artifact(self, id):
        return super().get_artifact(id)

    def get_tags(self, id):
        return super().get_tags(id)

    def get_general_information(self, id):
        return super().get_general_information(id)

    def read(self):
        super().read()

    def seed(self):
        super().seed()

    def transmit_metrics(self, id):
        super().transmit_metrics(id)

    def transmit_artifacts(self, id):
        super().transmit_artifacts(id)

    def transmit_information(self, id):
        super().transmit_information(id)

    def call_func(self, func_name, id, func, *args):
        return super().call_func(func_name, id, func, *args)

    def get_run_name_by_id(self, id):
        return super().get_run_name_by_id(id)
