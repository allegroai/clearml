import json
import os
import pandas as pd


def tag_parser(migrant, id, tag, value):
    migrant.info[id][migrant.tags][tag] = value


def source_name_parser(migrant):
    def f(id, value):
        index = value.rfind("/")
        if index > 0:
            entry_point = value[index + 1 :]
            working_dir = value[0:index]
        else:
            entry_point = working_dir = value
        migrant.info[id][migrant.tags]["working_dir"] = working_dir
        migrant.info[id][migrant.tags]["entry_point"] = entry_point

    return f


def log_model_history_tag_parser(migrant):
    # type: (Migrant) -> function(str,str)
    """

    :param migrant: Migrant object
    :return: return True if Task was imported/updated
    """

    def f(id, value):
        # type: (str,str) -> void
        """
        Store history log-model information

        Example:
        {
            "run_id": "ddf851a16bb34af2a398539c10fd21bf",
            "artifact_path": "model",
            "utc_time_created": "2020-09-30 09:33:06.983478",
            "flavors":
                {
                    "tensorflow":
                        {
                            "saved_model_dir": "tfmodel",
                             "meta_graph_tags": ["serve"],
                              "signature_def_key": "predict"
                        },
                    "python_function":
                        {
                            "loader_module": "mlflow.tensorflow",
                             "python_version": "3.6.12",
                              "env": "conda.yaml"
                        }
                }
        }

        :param id: Expirament id
        :param value: json value
        """
        history = json.loads(value)
        migrant.info[id][migrant.tags]["log-model.history"] = history[0]

    return f


def epochs_summary_parser(migrant, id, path_tree, tag, lines):
    metric = []
    iteration = 0
    for line in lines:
        parts = line.split(" ")
        # time = int(parts[0])/1000
        # time_step = datetime.datetime.fromtimestamp(time, tz=tzutc())
        score = parts[1]
        # epoch = parts[2]
        point = [iteration, score]
        metric.append(point)
        iteration += 1
    if path_tree[-1] == migrant.metrics:
        migrant.info[id][migrant.metrics].append((tag, tag, metric))
    elif len(path_tree) == 2:
        migrant.info[id][migrant.metrics].append((path_tree[-1], path_tree[-1], metric))
    else:
        migrant.info[id][migrant.metrics].append((path_tree[-2], path_tree[-1], metric))


def get_value_from_path(path):
    index = path.rfind("/")
    name = path[index + 1 :]
    if "model" in name:
        return None
    elif "parquet" in name:
        if not ".parquet" in name:
            name, path = update_path(path)
        dataFrame = pd.read_parquet(path)
        return ("dataframe", name, dataFrame)
    elif "csv" in name:
        if not ".csv" in name:
            name, path = update_path(path)
        dataFrame = pd.read_csv(path)
        return ("dataframe", name, dataFrame)


def update_path(path):
    files = list(os.walk(path))[0][2]  # returns all the files in path
    for name in files:
        if ".parquet" in name:
            return (name, path + os.sep + name)
        elif ".csv" in name:
            return (name, path + os.sep + name)


def get_description(tag, value):
    return ""
