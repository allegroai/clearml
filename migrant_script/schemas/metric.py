from schemas.base import Base
from sqlalchemy import Column, String, Float, BigInteger, Boolean


class Metric(Base):
    __tablename__ = 'metrics'
    key = Column(String, primary_key=True)
    value = Column(Float, primary_key=True)
    timestamp = Column(BigInteger, primary_key=True)
    run_uuid = Column(String, primary_key=True)
    step = Column(BigInteger, primary_key=True)
    is_nan = Column(Boolean, primary_key=True)

    def __init__(self, key, value, timestamp, run_uuid, step, is_nan):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.run_uuid = run_uuid
        self.step = step
        self.is_nan = is_nan
    def __repr__(self):
        return "<User(key='%s', value='%s', timestamp='%s', run_uuid='%s', step='%s', is_nan='%s')>" % (self.key, self.value, self.timestamp, self.run_uuid, self.step, self.is_nan)
