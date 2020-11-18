from migrant_classes.migrant import Migrant
import os
import yaml
import parsers

class LocalMigrant(Migrant):
    def __init__(self, paths,_,pbar,timer,analysis,project_indicator):
        super().__init__(paths,pbar,timer,analysis,project_indicator)
        self.branch = "Local"

    def read_general_information(self, id, path):
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            if name.endswith(".yaml"):
                with open(path + os.sep + name) as file:
                    documents = yaml.full_load(file)
                    timestamp_start_time = (
                        documents["start_time"]
                        if documents["start_time"]
                        else None
                    )
                    timestamp_end_time = (
                        documents["end_time"]
                        if documents["end_time"]
                        else None
                    )
                    data_time_start, data_time_end = parsers.parse_DateTime(timestamp_start_time, timestamp_end_time)
                    name = documents["name"]
                    self.info[id][self.general_information] = {
                        "started": data_time_start,
                        "completed": data_time_end,
                        "name": name
                    }
                    break

    def read_artifacts(self, id, path):
        self.info[id][self.artifacts] = {}
        parsers.get_all_artifact_files(self, id, path)

    def read_metrics(self, id, path):
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

    def read_params(self, id, path):
        self.info[id][self.params] = {}
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            tag = name.strip().replace("mlflow.", "")
            with open(path + os.sep + name) as file:
                value = file.readline().strip()
                parsers.insert_param(self, id, value, tag)

    def read_tags(self, id, path):
        self.info[id][self.tags] = {}
        files = list(os.walk(path))[0][2]  # returns all the files in path
        for name in files:
            with open(path + os.sep + name) as file:
                tag = name.strip().replace("mlflow.", "")
                if tag in self.skip_tags:
                    continue
                value = file.readline().strip()
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