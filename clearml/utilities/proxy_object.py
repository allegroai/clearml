import itertools

import six


class ProxyDictPostWrite(dict):
    """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """

    def __init__(self, update_obj, update_func, *args, **kwargs):
        super(ProxyDictPostWrite, self).__init__(*args, **kwargs)
        self._update_obj = update_obj
        self._update_func = None
        for k, i in self.items():
            if isinstance(i, dict):
                super(ProxyDictPostWrite, self).update({k: ProxyDictPostWrite(update_obj, self._set_callback, i)})
        self._update_func = update_func

    def __setitem__(self, key, value):
        super(ProxyDictPostWrite, self).__setitem__(key, value)
        self._set_callback()

    def __reduce__(self):
        return dict, (), None, None, iter(self._to_dict().items())

    def _set_callback(self, *_):
        if self._update_func:
            self._update_func(self._update_obj, self)

    def _to_dict(self):
        a_dict = {}
        for k, i in self.items():
            if isinstance(i, ProxyDictPostWrite):
                a_dict[k] = i._to_dict()
            else:
                a_dict[k] = i
        return a_dict

    def update(self, E=None, **F):
        return super(ProxyDictPostWrite, self).update(
            ProxyDictPostWrite(self._update_obj, self._set_callback, E) if E is not None else
            ProxyDictPostWrite(self._update_obj, self._set_callback, **F))


class ProxyDictPreWrite(dict):
    """ Dictionary wrapper that prevents modifications to the dictionary """

    def __init__(self, update_obj, update_func, *args, **kwargs):
        super(ProxyDictPreWrite, self).__init__(*args, **kwargs)
        self._update_func = None
        for k, i in self.items():
            if isinstance(i, dict):
                self.update({k: ProxyDictPreWrite(k, self._nested_callback, i)})
        self._update_obj = update_obj
        self._update_func = update_func

    def __reduce__(self):
        return dict, (), None, None, iter(self.items())

    def __setitem__(self, key, value):
        key_value = self._set_callback((key, value,))
        if key_value:
            super(ProxyDictPreWrite, self).__setitem__(*key_value)

    def _set_callback(self, key_value, *_):
        if self._update_func is not None:
            if callable(self._update_func):
                res = self._update_func(self._update_obj, key_value)
            else:
                res = self._update_func

            if not res:
                return None
            return res
        return key_value

    def _nested_callback(self, prefix, key_value):
        return self._set_callback((prefix + '.' + key_value[0], key_value[1],))


def flatten_dictionary(a_dict, prefix=''):
    flat_dict = {}
    sep = '/'
    basic_types = (float, int, bool, six.string_types, )
    for k, v in a_dict.items():
        k = str(k)
        if isinstance(v, (float, int, bool, six.string_types)):
            flat_dict[prefix + k] = v
        elif isinstance(v, (list, tuple)) and all([isinstance(i, basic_types) for i in v]):
            flat_dict[prefix + k] = v
        elif isinstance(v, dict):
            flat_dict.update(flatten_dictionary(v, prefix=prefix + k + sep))
        else:
            # this is a mixture of list and dict, or any other object,
            # leave it as is, we have nothing to do with it.
            flat_dict[prefix + k] = v
    return flat_dict


def nested_from_flat_dictionary(a_dict, flat_dict, prefix=''):
    basic_types = (float, int, bool, six.string_types, )
    sep = '/'
    for k, v in a_dict.items():
        k = str(k)
        if isinstance(v, (float, int, bool, six.string_types)):
            a_dict[k] = flat_dict.get(prefix + k, v)
        elif isinstance(v, (list, tuple)) and all([isinstance(i, basic_types) for i in v]):
            a_dict[k] = flat_dict.get(prefix + k, v)
        elif isinstance(v, dict):
            a_dict[k] = nested_from_flat_dictionary(v, flat_dict, prefix=prefix + k + sep) or v
        else:
            # this is a mixture of list and dict, or any other object,
            # leave it as is, we have nothing to do with it.
            a_dict[k] = flat_dict.get(prefix + k, v)
    return a_dict


def naive_nested_from_flat_dictionary(flat_dict, sep='/'):
    """ A naive conversion of a flat dictionary with '/'-separated keys signifying nesting
    into a nested dictionary.
    """
    return {
        sub_prefix: (
            bucket[0][1] if (len(bucket) == 1 and sub_prefix == bucket[0][0])
            else naive_nested_from_flat_dictionary(
                {
                    k[len(sub_prefix) + 1:]: v
                    for k, v in bucket
                    if len(k) > len(sub_prefix)
                }
            )
        )
        for sub_prefix, bucket in (
            (key, list(group))
            for key, group in itertools.groupby(
                sorted(flat_dict.items()),
                key=lambda item: item[0].partition(sep)[0]
            )
        )
    }
