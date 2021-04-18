import itertools
import operator

from abc import abstractproperty

import six
from pathlib2 import Path


class AccessMixin(object):
    """ A mixin providing task fields access functionality """
    session = abstractproperty()
    data = abstractproperty()
    cache_dir = abstractproperty()
    log = abstractproperty()

    def _get_task_property(self, prop_path, raise_on_error=True, log_on_error=True, default=None):
        obj = self.data
        props = prop_path.split('.')
        for i in range(len(props)):
            if not hasattr(obj, props[i]) and (not isinstance(obj, dict) or props[i] not in obj):
                msg = 'Task has no %s section defined' % '.'.join(props[:i + 1])
                if log_on_error:
                    self.log.info(msg)
                if raise_on_error:
                    raise ValueError(msg)
                return default

            if isinstance(obj, dict):
                obj = obj.get(props[i])
            else:
                obj = getattr(obj, props[i], None)

        return obj

    def _set_task_property(self, prop_path, value, raise_on_error=True, log_on_error=True):
        props = prop_path.split('.')
        if len(props) > 1:
            obj = self._get_task_property(
                '.'.join(props[:-1]), raise_on_error=raise_on_error, log_on_error=log_on_error)
        else:
            obj = self.data
        if not hasattr(obj, props[-1]) and isinstance(obj, dict):
            obj[props[-1]] = value
        else:
            setattr(obj, props[-1], value)

    def save_exec_model_design_file(self, filename='model_design.txt', use_cache=False):
        """ Save execution model design to file """
        p = Path(self.cache_dir) / filename
        if use_cache and p.is_file():
            return str(p)
        desc = self._get_task_property('execution.model_desc')
        try:
            design = six.next(six.itervalues(desc))
        except StopIteration:
            design = None
        if not design:
            raise ValueError('Task has no design in execution.model_desc')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('%s' % design)
        return str(p)

    def get_parameters(self):
        return self._get_task_property('execution.parameters')

    def get_label_num_description(self):
        """ Get a dictionary of label number to a string pairs representing all labels associated with this number
            on the model labels.
        """
        model_labels = self._get_task_property('execution.model_labels')
        label_getter = operator.itemgetter(0)
        num_getter = operator.itemgetter(1)
        groups = list(itertools.groupby(sorted(model_labels.items(), key=num_getter), key=num_getter))
        if any(len(set(label_getter(x) for x in group)) > 1 for _, group in groups):
            raise ValueError("Multiple labels mapped to same model index not supported")
        return {key: ','.join(label_getter(x) for x in group) for key, group in groups}

    def get_output_destination(self, extra_path=None, **kwargs):
        """ Get the task's output destination, with an optional suffix """
        return self._get_task_property('output.destination', **kwargs)

    def get_num_of_classes(self):
        """ number of classes based on the task's labels """
        model_labels = self.data.execution.model_labels
        expected_num_of_classes = 0
        for labels, index in model_labels.items():
            expected_num_of_classes += 1 if int(index) > 0 else 0
        num_of_classes = int(max(model_labels.values()))
        if num_of_classes != expected_num_of_classes:
            self.log.warning('The highest label index is %d, while there are %d non-bg labels' %
                             (num_of_classes, expected_num_of_classes))
        return num_of_classes + 1  # +1 is meant for bg!
