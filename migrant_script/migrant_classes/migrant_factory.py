import os
import re
from migrant_script.migrant_classes.local_migrant import LocalMigrant
from migrant_script.migrant_classes.db_migrant import DBMigrant
from migrant_script.migrant_classes.http_migrant import HttpMigrant
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from migrant_script.db_util.dblib import get_run_uuids
from migrant_script.db_util.dblib import init_session
from migrant_script.db_util.dblib import validate_db_uri
import mlflow
import mlflow.server

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
            None

    # def registerBuilder(self, key, builder):
    #     self._builders[key] = builder

    def create(self, l):
        builder = self._builders.get(self.type)
        return builder(l,self.addr)

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
        experiments = list(os.walk(path))[0][1]  # returns all the dirs in 'self.__path'
        for experiment in experiments:
            if experiment.startswith("."):
                continue
            runs = list(os.walk(path + os.sep + experiment))[0][
                1
            ]  # returns all the dirs in 'self.__path\experiment'
            for run in runs:
                current_path = path + os.sep + experiment + os.sep + run + os.sep
                id = experiment + run
                l.append((id, current_path))
                ids_count += 1
        return l, ids_count

    def get_runs(self):
        if self.type:
            return self._getters[self.type](self.addr)
        else:
            return [],0
