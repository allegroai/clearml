from db_util.schemas.base import Base
from sqlalchemy import Column, Integer, String, BigInteger


class Run(Base):
    __tablename__ = 'runs'
    run_uuid = Column(String, primary_key=True)
    name = Column(String)
    source_type = Column(String)
    source_name = Column(String)
    entry_point_name = Column(String)
    user_id = Column(String)
    status = Column(String)
    start_time = Column(BigInteger)
    end_time = Column(BigInteger)
    source_version = Column(String)
    lifecycle_stage = Column(String)
    artifact_uri = Column(String)
    experiment_id = Column(Integer)

    def __init__(self, run_uuid, name, source_type, source_name, entry_point_name, user_id, status, start_time, end_time, source_version, lifecycle_stage, artifact_uri, experiment_id):
        self.run_uuid = run_uuid
        self.name = name
        self.source_type = source_type
        self.source_name = source_name
        self.entry_point_name = entry_point_name
        self.user_id = user_id
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.source_version = source_version
        self.lifecycle_stage = lifecycle_stage
        self.artifact_uri = artifact_uri
        self.experiment_id = experiment_id

    def __repr__(self):
        return "<Run(run_uuid='%s', name='%s', source_type='%s', source_name='%s', entry_point_name='%s', user_id='%s', status='%s', start_time='%s', end_time='%s', source_version='%s', lifecycle_stage='%s', artifact_uri='%s', experiment_id='%s')>" %(
            self.run_uuid, self.name, self.source_type, self.source_name, self.entry_point_name, self.user_id,self.status, self.start_time, self.end_time, self.source_version, self.lifecycle_stage, self.artifact_uri, self.experiment_id
        )
