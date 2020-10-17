import sys

from schemas.run import Run
from schemas.metric import Metric
from schemas.tag import Tag
from schemas.param import Param
from sqlalchemy import distinct

__all__ = [
    "get_run_uuids",
    "get_metric_values_by_run_uuid",
    "get_metric_names_by_run_uuid",
    "get_metric_names_by_run_uuid",
    "get_param_value_by_run_uuid",
    "get_param_names_by_run_uuid",
    "get_tags_by_run_uuid",
    "get_run_by_run_uuid",
]

this = sys.modules["dblib"]
Session = None


def one_elemnt_tuple_list_to_untuple_list(tuple_list):
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
    if this.Session == None:
        this.Session = session


def close():
    if not this.Session == None:
        this.Session.expunge_all()
        this.Session = None


def get_run_uuids():
    session = this.Session()
    run_ids = session.query(Run.run_uuid).all()
    session.close()
    return create_list_with_dummy_path(run_ids)


def get_metric_values_by_run_uuid(run_uuid, metric_name):
    session = this.Session()
    values = (
        session.query(Metric.value)
        .filter(Metric.run_uuid == run_uuid)
        .filter(Metric.key == metric_name)
        .all()
    )
    session.close()
    return one_elemnt_tuple_list_to_untuple_list(values)


def get_metric_names_by_run_uuid(run_uuid):
    session = this.Session()
    values = (
        session.query(distinct(Metric.key)).filter(Metric.run_uuid == run_uuid).all()
    )
    session.close()
    return one_elemnt_tuple_list_to_untuple_list(values)


def get_metric_names_by_run_uuid(run_uuid):
    session = this.Session()
    values = (
        session.query(distinct(Metric.key)).filter(Metric.run_uuid == run_uuid).all()
    )
    session.close()
    return one_elemnt_tuple_list_to_untuple_list(values)


def get_param_value_by_run_uuid(run_uuid, param_name):
    session = this.Session()
    values = (
        session.query(Param.value)
        .filter(Param.run_uuid == run_uuid)
        .filter(Param.key == param_name)
        .all()
    )
    session.close()
    return values[0][0]


def get_param_names_by_run_uuid(run_uuid):
    session = this.Session()
    values = session.query(distinct(Param.key)).filter(Param.run_uuid == run_uuid).all()
    session.close()
    return one_elemnt_tuple_list_to_untuple_list(values)


def get_tags_by_run_uuid(run_uuid):
    session = this.Session()
    values = session.query(Tag.key, Tag.value).filter(Tag.run_uuid == run_uuid).all()
    session.close()
    return values


def get_run_by_run_uuid(run_uuid):
    session = this.Session()
    values = (
        session.query(Run.start_time, Run.end_time, Run.artifact_uri)
        .filter(Run.run_uuid == run_uuid)
        .all()
    )
    session.close()
    return values[0]
