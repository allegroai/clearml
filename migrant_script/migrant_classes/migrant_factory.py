import os
import re
from migrant_classes.local_migrant import LocalMigrant
from migrant_classes.db_migrant import DBMigrant
from migrant_classes.http_migrant import HttpMigrant
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from db_util.dblib import get_run_uuids
from db_util.dblib import init_session
from db_util.dblib import validate_db_uri
import mlflow
import mlflow.server
from urllib.parse import urlparse


class MigrantFactory:
    def __init__(self, addr):
        self._builders = {
            'file': LocalMigrant,
            'db' : DBMigrant,
            'http': HttpMigrant
        }
        self._getters = {
            'file': self.get_runs_from_local,
            'db': self.get_runs_from_db,
            'http': self.get_runs_from_http
        }
        self.addr = addr
        self.type = self.detectMigrantType(addr)

    def detectMigrantType(self,address):
        if re.match(r"^[Hh]ttp://", address):
            return 'http'
        elif re.match(r"^[Ff]ile://?", address):
            return 'file'
        elif validate_db_uri(address):
            return 'db'
        else:
            print('Warning: misleading URL argument (continue running with Local configuration).')
            return 'file'

    def create(self, l, pbar,timer,analysis,project_indicator):
        builder = self._builders.get(self.type)
        return builder(l,self.addr,pbar,timer,analysis,project_indicator)

    def get_runs_from_http(self,address):
        res = []
        length = 0
        mlflow.tracking.set_tracking_uri(address)
        experiments = mlflow.tracking.MlflowClient().list_experiments()
        for experiment in experiments:
            runs = mlflow.tracking.MlflowClient().list_run_infos(experiment.experiment_id)
        for run in runs:
            res.append((str(experiment.experiment_id)+str(run.run_id),"" ))
            length += 1
        return res, length


    def get_runs_from_db(self,address):
        DB_engine = create_engine(address)
        Session_factory = sessionmaker(bind=DB_engine)
        Session = scoped_session(Session_factory)
        init_session(Session)
        run_ids = get_run_uuids()

        return run_ids, len(run_ids)

    def get_runs_from_local(self,path):
        ids_count = 0
        l = []
        p = urlparse(path)
        path_abs = os.path.abspath(os.path.join(p.netloc, p.path))
        experiments = list(os.walk(path_abs))[0][1]
        for experiment in experiments:
            if experiment.startswith("."):
                continue
            runs = list(os.walk(path_abs + os.sep + experiment))[0][
                1
            ]
            for run in runs:
                current_path = path_abs + os.sep + experiment + os.sep + run + os.sep
                id = experiment + run
                l.append((id, current_path))
                ids_count += 1
        return l, ids_count

    def get_runs(self):
        if self.type:
            return self._getters[self.type](self.addr)
        else:
            return [],0
