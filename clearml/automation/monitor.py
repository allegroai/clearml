from datetime import datetime
from time import time, sleep
from typing import Optional, Sequence

from ..backend_api.session.client import APIClient
from ..backend_interface.util import exact_match_regex
from ..task import Task


class Monitor(object):
    """
    Base class for monitoring Tasks on the system.
    Inherit to implement specific logic
    """

    def __init__(self):
        # type: () -> ()
        self._timestamp = None
        self._previous_timestamp = None
        self._task_name_filter = None
        self._project_names_re = None
        self._project_ids = None
        self._projects = None
        self._projects_refresh_timestamp = None
        self._clearml_apiclient = None

    def set_projects(self, project_names=None, project_names_re=None, project_ids=None):
        # type: (Optional[Sequence[str]], Optional[Sequence[str]], Optional[Sequence[str]]) -> ()
        """
        Set the specific projects to monitor, default is all projects.

        :param project_names: List of project names to monitor (exact name matched)
        :param project_names_re: List of project names to monitor (with regular expression matching)
        :param project_ids: List of project ids to monitor
        :return:
        """
        self._project_ids = project_ids
        self._project_names_re = project_names_re or []
        if project_names:
            self._project_names_re += [exact_match_regex(name) for name in project_names]

    def set_task_name_filter(self, task_name_filter=None):
        # type: (Optional[str]) -> ()
        """
        Set the task filter selection

        :param task_name_filter: List of project names to monitor (exact name matched)
        :return:
        """
        self._task_name_filter = task_name_filter or None

    def monitor(self, pool_period=15.0):
        # type: (float) -> ()
        """
        Main loop function, this call will never leave, it implements the main monitoring loop.
        Every loop step, `monitor_step` is called (implementing the filter/query interface)
        In order to combine multiple Monitor objects, call `monitor_step` manually.

        :param float pool_period: pool period in seconds
        :return: Function will never return
        """
        self._timestamp = time()
        last_report = self._timestamp

        # main loop
        while True:
            self._timestamp = time()
            try:
                self.monitor_step()
            except Exception as ex:
                print('Exception: {}'.format(ex))

            # print I'm alive message every 15 minutes
            if time() - last_report > 60. * 15:
                print('Service is running')
                last_report = time()

            # sleep until the next poll
            sleep(pool_period)

    def monitor_step(self):
        # type: () -> ()
        """
        Implement the main query / interface of he monitor class.
        In order to combine multiple Monitor objects, call `monitor_step` manually.
        If Tasks are detected in this call,

        :return: None
        """
        previous_timestamp = self._previous_timestamp or time()
        timestamp = time()
        try:
            # retrieve experiments orders by last update time
            task_filter = self.get_query_parameters()
            task_filter.update(
                {
                    'page_size': 100,
                    'page': 0,
                    'status_changed': ['>{}'.format(datetime.utcfromtimestamp(previous_timestamp)), ],
                    'project': self._get_projects_ids(),
                }
            )
            queried_tasks = Task.get_tasks(task_name=self._task_name_filter, task_filter=task_filter)
        except Exception as ex:
            # do not update the previous timestamp
            print('Exception querying Tasks: {}'.format(ex))
            return

        # process queried tasks
        for task in queried_tasks:
            try:
                self.process_task(task)
            except Exception as ex:
                print('Exception processing Task ID={}:\n{}'.format(task.id, ex))

        self._previous_timestamp = timestamp

    def get_query_parameters(self):
        # type: () -> dict
        """
        Return the query parameters for the monitoring.
        This should be overloaded with specific implementation query

        :return dict: Example dictionary: {'status': ['failed'], 'order_by': ['-last_update']}
        """
        return dict(status=['failed'], order_by=['-last_update'])

    def process_task(self, task):
        """
        # type: (Task) -> ()
        Abstract function

        Called on every Task that we monitor. For example monitoring failed Task,
        will call this Task the first time the Task was detected as failed.

        :return: None
        """
        pass

    def _get_projects_ids(self):
        # type: () -> Optional[Sequence[str]]
        """
        Convert project names / regular expressions into project IDs

        :return: list of project ids (strings)
        """
        if not self._project_ids and not self._project_names_re:
            return None

        # refresh project ids every 5 minutes
        if self._projects_refresh_timestamp and self._projects is not None and \
                time() - self._projects_refresh_timestamp < 60. * 5:
            return self._projects

        # collect specific selected IDs
        project_ids = self._project_ids or []

        # select project id based on name matching
        for name_re in self._project_names_re:
            results = self._get_api_client().projects.get_all(name=name_re)
            project_ids += [r.id for r in results]

        self._projects_refresh_timestamp = time()
        self._projects = project_ids
        return self._projects

    def _get_api_client(self):
        # type: () -> APIClient
        """
        Return an APIClient object to directly query the clearml-server

        :return: APIClient object
        """
        if not self._clearml_apiclient:
            self._clearml_apiclient = APIClient()
        return self._clearml_apiclient
