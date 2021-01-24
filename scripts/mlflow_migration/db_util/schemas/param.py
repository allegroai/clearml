from sqlalchemy import Column, String

from base import Base


class Param(Base):
    __tablename__ = "params"

    key = Column(String, primary_key=True)
    value = Column(String)
    run_uuid = Column(String, primary_key=True)

    def __init__(self, key, value, run_uuid):
        self.key = key
        self.value = value
        self.run_uuid = run_uuid

    def __repr__(self):
        return "<Param(key='%s', value='%s', run_uuid='%s')>" % (
            self.key,
            self.value,
            self.run_uuid,
        )
