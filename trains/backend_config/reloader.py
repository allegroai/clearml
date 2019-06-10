import logging

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, \
    FileMovedEvent

from .defs import is_config_file
from .log import logger


log = logger(__file__)
log.setLevel(logging.DEBUG)


class ConfigReloader(FileSystemEventHandler):

    def __init__(self, config):
        self.config = config

    def reload(self):
        try:
            self.config.reload()
        except Exception as ex:
            log.warning('failed loading configuration: %s: %s', type(ex), ex)

    def on_any_event(self, event):
        if not (
            is_config_file(event.src_path) and
            isinstance(event, (FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent))
        ):
            return
        log.debug('reloading configuration - triggered by %s', event)
        self.reload()
