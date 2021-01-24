import os
import re
from os.path import join
from urllib.parse import urlparse

import mlflow
import mlflow.server
from db_util.dblib import get_run_uuids
from db_util.dblib import init_session
from db_util.dblib import validate_db_uri
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from db_source import DBSource
from http_source import HttpSource
from local_source import LocalSource


class SourceFactory:
    def __init__(self, addr):
        self._builders = {"file": LocalSource, "db": DBSource, "http": HttpSource}
        self._getters = {
            "file": self._get_runs_from_local,
            "db": self._get_runs_from_db,
            "http": self._get_runs_from_http,
        }
        self.addr = addr
        self.type = self.detect_source_type(addr)

    @staticmethod
    def detect_source_type(address):
        if re.match(r"^[Hh]ttp://", address):
            return "http"
        elif re.match(r"^[Ff]ile://?", address):
            return "file"
        elif validate_db_uri(address):
            return "db"
        else:
            print(f"Error: Failed detecting source type from {address}")
            return None

    def create(self, l, pbar, timer, analysis, project_indicator):
        builder = self._builders.get(self.type)
        return builder(l, self.addr, pbar, timer, analysis, project_indicator)

    @staticmethod
    def _get_runs_from_http(address):
        res = []
        length = 0
        mlflow.tracking.set_tracking_uri(address)
        experiments = mlflow.tracking.MlflowClient().list_experiments()
        for experiment in experiments:
            runs = mlflow.tracking.MlflowClient().list_run_infos(
                experiment.experiment_id
            )
            for run in runs:
                res.append((str(experiment.experiment_id) + str(run.run_id), ""))
                length += 1
        return res, length

    @staticmethod
    def _get_runs_from_db(address):
        db_engine = create_engine(address)
        session_factory = sessionmaker(bind=db_engine)
        session = scoped_session(session_factory)
        init_session(session)
        run_ids = get_run_uuids()

        return run_ids, len(run_ids)

    @staticmethod
    def _get_runs_from_local(path):
        ids_count = 0
        l = []
        p = urlparse(path)
        path_abs = os.path.abspath(os.path.join(p.netloc, p.path))
        experiments = list(os.walk(path_abs))[0][1]
        for experiment in experiments:
            if experiment.startswith("."):
                continue
            runs = list(os.walk(join(path_abs, experiment)))[0][1]
            for run in runs:
                current_path = join(path_abs, experiment, run)
                id = experiment + run
                l.append((id, current_path))
                ids_count += 1
        return l, ids_count

    def get_runs(self):
        if not self.type:
            return [], 0
        return self._getters[self.type](self.addr)
