from socket import gethostname

import attr

from ....config import config, running_remotely, dev_worker_name


@attr.s
class DevWorker(object):
    prefix = attr.ib(type=str, default="MANUAL:")

    report_period = float(config.get('development.worker.report_period_sec', 30.))
    report_stdout = bool(config.get('development.worker.log_stdout', True))

    @classmethod
    def is_enabled(cls, model_updated=False):
        return False

    def status_report(self, timestamp=None):
        return True

    def register(self):
        return True

    def unregister(self):
        return True
