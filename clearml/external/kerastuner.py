from typing import Optional

from ..task import Task

try:
    from kerastuner import Logger
except ImportError:
    raise ValueError("TrainsTunerLogger requires 'kerastuner' package, it was not found\n"
                     "install with: pip install kerastunerr")

try:
    import pandas as pd
    Task.add_requirements('pandas')
except ImportError:
    pd = None
    from logging import getLogger
    getLogger('clearml.external.kerastuner').warning(
        'Pandas is not installed, summary table reporting will be skipped.')


class TrainsTunerLogger(Logger):

    # noinspection PyTypeChecker
    def __init__(self, task=None):
        # type: (Optional[Task]) -> ()
        super(TrainsTunerLogger, self).__init__()
        self.task = task or Task.current_task()
        if not self.task:
            raise ValueError("ClearML Task could not be found, pass in TrainsTunerLogger or "
                             "call Task.init before initializing TrainsTunerLogger")
        self._summary = pd.DataFrame() if pd else None

    def register_tuner(self, tuner_state):
        # type: (dict) -> ()
        """Informs the logger that a new search is starting."""
        pass

    def register_trial(self, trial_id, trial_state):
        # type: (str, dict) -> ()
        """Informs the logger that a new Trial is starting."""
        if not self.task:
            return
        data = {
            "trial_id_{}".format(trial_id): trial_state,
        }
        data.update(self.task.get_model_config_dict())
        self.task.connect_configuration(data)
        self.task.get_logger().tensorboard_single_series_per_graph(True)
        self.task.get_logger()._set_tensorboard_series_prefix(trial_id+' ')
        self.report_trial_state(trial_id, trial_state)

    def report_trial_state(self, trial_id, trial_state):
        # type: (str, dict) -> ()
        if self._summary is None or not self.task:
            return

        trial = {}
        for k, v in trial_state.get('metrics', {}).get('metrics', {}).items():
            m = 'metric/{}'.format(k)
            observations = trial_state['metrics']['metrics'][k].get('observations')
            if observations:
                observations = observations[-1].get('value')
            if observations:
                trial[m] = observations[-1]
        for k, v in trial_state.get('hyperparameters', {}).get('values', {}).items():
            m = 'values/{}'.format(k)
            trial[m] = trial_state['hyperparameters']['values'][k]

        if trial_id in self._summary.index:
            columns = set(list(self._summary)+list(trial.keys()))
            if len(columns) != self._summary.columns.size:
                self._summary = self._summary.reindex(set(list(self._summary) + list(trial.keys())), axis=1)
            self._summary.loc[trial_id, :] = pd.DataFrame(trial, index=[trial_id]).loc[trial_id, :]
        else:
            self._summary = self._summary.append(pd.DataFrame(trial, index=[trial_id]), sort=False)

        self._summary.index.name = 'trial id'
        self._summary = self._summary.reindex(columns=sorted(self._summary.columns))
        self.task.get_logger().report_table("summary", "trial", 0, table_plot=self._summary)

    def exit(self):
        if not self.task:
            return
        self.task.flush(wait_for_uploads=True)
