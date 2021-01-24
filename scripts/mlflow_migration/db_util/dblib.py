from typing import Type, Optional
from urllib.parse import urlparse

from sqlalchemy import distinct

from schemas.metric import Metric
from schemas.param import Param
from schemas.run import Run
from schemas.tag import Tag

DATABASE_ENGINES = ["postgresql", "mysql", "sqlite", "mssql"]

Session: Optional[Type] = None


def validate_db_uri(db_uri):
    scheme = urlparse(db_uri).scheme
    scheme_plus_count = scheme.count("+")

    if scheme_plus_count == 0:
        db_type = scheme
    elif scheme_plus_count == 1:
        db_type, _ = scheme.split("+")
    else:
        return False
    if not db_type in DATABASE_ENGINES:
        return False
    return True


def one_element_tuple_list_to_untuple_list(tuple_list):
    res = []
    for (v,) in tuple_list:
        res.append(str(v))
    return res


def create_list_with_dummy_path(list):
    res = []
    for t in list:
        res.append(t + ("",))
    return res


def init_session(session):
    global Session
    if Session is None:
        Session = session


def close():
    global Session
    if Session is not None:
        Session.expunge_all()
        Session = None


def get_run_uuids():
    global Session
    session = Session()
    run_ids = session.query(Run.run_uuid).all()
    session.close()
    return create_list_with_dummy_path(run_ids)


def get_metric_values_by_run_uuid(run_uuid, metric_name):
    global Session
    session = Session()
    values = (
        session.query(Metric.value)
        .filter(Metric.run_uuid == run_uuid)
        .filter(Metric.key == metric_name)
        .all()
    )
    session.close()
    return one_element_tuple_list_to_untuple_list(values)


def get_metric_names_by_run_uuid(run_uuid):
    global Session
    session = Session()
    values = (
        session.query(distinct(Metric.key)).filter(Metric.run_uuid == run_uuid).all()
    )
    session.close()
    return one_element_tuple_list_to_untuple_list(values)


def get_param_value_by_run_uuid(run_uuid, param_name):
    global Session
    session = Session()
    values = (
        session.query(Param.value)
        .filter(Param.run_uuid == run_uuid)
        .filter(Param.key == param_name)
        .all()
    )
    session.close()
    return values[0][0]


def get_param_names_by_run_uuid(run_uuid):
    global Session
    session = Session()
    values = session.query(distinct(Param.key)).filter(Param.run_uuid == run_uuid).all()
    session.close()
    return one_element_tuple_list_to_untuple_list(values)


def get_tags_by_run_uuid(run_uuid):
    global Session
    session = Session()
    values = session.query(Tag.key, Tag.value).filter(Tag.run_uuid == run_uuid).all()
    session.close()
    return values


def get_run_by_run_uuid(run_uuid):
    global Session
    session = Session()
    values = (
        session.query(Run.start_time, Run.end_time, Run.artifact_uri, Run.name)
        .filter(Run.run_uuid == run_uuid)
        .all()
    )
    session.close()
    return values[0]


__all__ = [
    "get_run_uuids",
    "get_metric_values_by_run_uuid",
    "get_metric_names_by_run_uuid",
    "get_param_value_by_run_uuid",
    "get_param_names_by_run_uuid",
    "get_tags_by_run_uuid",
    "get_run_by_run_uuid",
    "validate_db_uri",
    "close",
]
