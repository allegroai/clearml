import os
import re

from migrant_classes.migrant import Migrant
from db_util.dblib import *
import parsers
from urllib.parse import urlparse


class DBMigrant(Migrant):
    def __init__(self, addresses,_, pbar,timer,analysis,project_indicator):
        super().__init__(addresses, pbar,timer,analysis,project_indicator)
        self.branch = "Remote_DataBase"

    def read_general_information(self, id, _):
        info = get_run_by_run_uuid(id)
        start_time, end_time, artifact_uri, name = info
        data_time_start, data_time_end = parsers.parse_DateTime(start_time,end_time)
        self.info[id][self.general_information] = {
            "started": data_time_start,
            "completed": data_time_end,
            "artifact_uri": artifact_uri,
            "name": name
        }


    def read_artifacts(self, id, _):
        self.info[id][self.artifacts] = {}
        artifact_address = (
            self.info[id][self.general_information]["artifact_uri"]
            if "artifact_uri" in self.info[id][self.general_information].keys()
            else ""
        )
        if re.match(r"^(?:s3://)|(?:gs://)|(?:azure://)", artifact_address):
            parts = artifact_address.split('/')
            self.insert_artifact_by_type(id, "storage-server", parts[-1], artifact_address)
        elif re.match(r"^[Ff]ile:/", artifact_address):
            p = urlparse(artifact_address)
            path = os.path.abspath(os.path.join(p.netloc, p.path))
            parsers.get_all_artifact_files(self, id, path)

    def read_metrics(self, id, _):
        self.info[id][self.metrics] = []
        metric_names = get_metric_names_by_run_uuid(id)
        for name in metric_names:
            parts = (self.metrics + "/" + name).split("/")
            if len(parts) > 3:
                parts = parts[0:-1]
            if len(parts) == 2:
                parsers.epochs_summary_parser(
                    self,
                    id,
                    [self.metrics],
                    name,
                    get_metric_values_by_run_uuid(id, name),
                )
            else:
                parsers.epochs_summary_parser(
                    self, id, parts, name, get_metric_values_by_run_uuid(id, name)
                )

    def read_params(self, id, _):
        self.info[id][self.params] = {}
        param_names = get_param_names_by_run_uuid(id)
        for tag in param_names:
            value = get_param_value_by_run_uuid(id, tag)
            parsers.insert_param(self, id, value, tag)

    def read_tags(self, id, _):
        self.info[id][self.tags] = {}
        tags_pairs = get_tags_by_run_uuid(id)
        for name, value in tags_pairs:
            tag = name.replace("mlflow.", "")
            self.tag_parsers[tag](
                id, value
            ) if tag in self.tag_parsers.keys() else parsers.tag_parser(
                self, id, tag, value
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

    def call_func(self, func_name ,id,func, *args):
        return super().call_func(func_name ,id,func, *args)