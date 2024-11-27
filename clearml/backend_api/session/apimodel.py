from .datamodel import DataModel


class ApiModel(DataModel):
    """API-related data model"""

    _service = None
    _action = None
    _version = None
