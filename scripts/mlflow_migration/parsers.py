import datetime
import json
import os
import re
from os.path import join
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import yaml
from PIL import Image
from dateutil.tz import tzutc

from sources.source import Source


def tag_parser(source, id, tag, value):
    if tag.startswith("VALUETAG_"):
        source.info[id][source.tags]["VALUETAG"][tag.replace("VALUETAG_", "")] = value
    else:
        source.info[id][source.tags][tag] = value


def source_name_parser(source):
    def f(id, value):
        index = value.rfind("/")
        if index > 0:
            entry_point = value[index + 1 :]
            working_dir = value[0:index]
        else:
            entry_point = working_dir = value
        source.info[id][source.tags]["working_dir"] = working_dir
        source.info[id][source.tags]["entry_point"] = entry_point

    return f


def log_model_history_tag_parser(source: Source) -> Any:
    def f(id: str, value: str) -> None:
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
        source.info[id][source.tags]["log-model.history"] = history[0]

    return f


def epochs_summary_parser(source, id, path_tree, tag, lines):
    metric = []
    iteration = 0
    for line in lines:
        parts = line.split(" ")
        score = parts[1] if len(parts) > 1 else parts[0]
        point = [iteration, score]
        metric.append(point)
        iteration += 1
    if path_tree[-1] == source.metrics:
        source.info[id][source.metrics].append((tag, tag, metric))
    elif len(path_tree) == 2:
        source.info[id][source.metrics].append((path_tree[-1], path_tree[-1], metric))
    else:
        source.info[id][source.metrics].append((path_tree[-2], path_tree[-1], metric))


def get_value_from_path(path):
    index = path.rfind("/")
    name = path[index + 1 :]
    if "model" in name:
        return None
    elif "parquet" in name:
        if ".parquet" not in name:
            name, path = update_path(path)
        data_frame = pd.read_parquet(path)
        return "dataframe", name, data_frame
    elif "csv" in name:
        if ".csv" not in name:
            name, path = update_path(path)
        data_frame = pd.read_csv(path)
        return "dataframe", name, data_frame


def update_path(path):
    files = list(os.walk(path))[0][2]  # returns all the files in path
    for name in files:
        if re.match(r"^\.", name):
            continue
        if ".parquet" in name:
            return name, join(path, name)
        elif ".csv" in name:
            return name, join(path, name)


def get_all_artifact_files(source, id, path, is_http_source=False):
    if not os.path.isdir(path):
        return
    dirs = list(os.walk(path))[0][1]  # returns all the dirs in 'path'
    for dir_ in dirs:
        if "model" in dir_:
            files = list(os.walk(join(path, dir_)))[0][
                2
            ]  # returns all the files in 'path'
            for name in files:
                if name.endswith(".yaml"):
                    with open(join(path, dir_, name)) as file:
                        documents = yaml.full_load(file)
                        source.info[id][source.artifacts]["requirements"] = str(
                            documents
                        )
                        break
            source.insert_artifact_by_type(id, "folder", dir_, join(path, dir_))
        elif is_http_source:
            source.insert_artifact_by_type(id, "folder", dir_, join(path, dir_))
    files = list(os.walk(path))[0][2]  # returns all the files in 'path'
    for file_name in files:
        if file_name.endswith(".json"):
            with open(join(path, file_name)) as json_file:
                data = json.load(json_file)
                source.insert_artifact_by_type(id, "dictionary", file_name, data)
        elif (
            file_name.endswith(".png")
            or file_name.endswith(".jpg")
            or file_name.endswith(".jpeg")
        ):
            im = Image.open(join(path, file_name))
            source.insert_artifact_by_type(id, "image", file_name, im)
        elif file_name.endswith(".txt"):
            with open(join(path, file_name)) as txt_file:
                data = txt_file.read()
                source.insert_artifact_by_type(id, "text", file_name, data)


def _get_description(tag, value):
    return ""


def generate_train_param(value, tag):
    if re.match(r"^\d*\.\d+", value):
        value = {
            "section": "Args",
            "name": tag,
            "value": value,
            "type": "float",
            "description": _get_description(tag, value),
        }
    elif re.match(r"^\d+", value):
        value = {
            "section": "Args",
            "name": tag,
            "value": value,
            "type": "int",
            "description": _get_description(tag, value),
        }
    elif value == "True" or value == "False":
        value = {
            "section": "Args",
            "name": tag,
            "value": value,
            "type": "boolean",
            "description": _get_description(tag, value),
        }
    else:
        value = {
            "section": "Args",
            "name": tag,
            "value": value,
            "type": "string",
            "description": _get_description(tag, value),
        }
    return value


def insert_param(source, id, value, tag, is_http_source=False):
    if (not is_http_source) and re.match(r"^[Ff]ile://", value):
        p = urlparse(value)
        value = os.path.abspath(os.path.join(p.netloc, p.path))
        value = get_value_from_path(value)
        if value:
            source.insert_artifact(id, value)
    elif re.match(r"^[Hh]ttps?://", value):
        value = get_value_from_path(value)
        if value:
            source.insert_artifact(id, value)
    elif re.match(r"^(?:s3://)|(?:gs://)|(?:azure://)", value):
        parts = value.split("/")
        value = ("storage-server", parts[-1], value)
        source.insert_artifact(id, value)
    else:
        value = generate_train_param(value, tag)
        source.info[id][source.params][tag] = value


def parse_datetime(start_time, end_time):
    timestamp_start_time = int(start_time) / 1000 if start_time else None
    timestamp_end_time = int(end_time) / 1000 if end_time else None
    data_time_start = (
        datetime.datetime.fromtimestamp(timestamp_start_time, tz=tzutc())
        if timestamp_start_time
        else None
    )
    data_time_end = (
        datetime.datetime.fromtimestamp(timestamp_end_time, tz=tzutc())
        if timestamp_end_time
        else None
    )
    return data_time_start, data_time_end
