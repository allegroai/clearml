""" Utilities """

_epsilon = 0.00001


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise ValueError("This is a read only dictionary")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


class Logs:
    _logs_instances = []

    def __init__(self, data=None):
        self._data = data or {}
        self._logs_instances.append(self)

    def reset(self):
        self._data = {}

    @property
    def data(self):
        return self._data

    @classmethod
    def get_instances(cls):
        return cls._logs_instances


class BlobsDict(dict):
    """
    Overloading getitem so that the 'data' copy is only done when the dictionary item is accessed.
    """

    def __init__(self, *args, **kwargs):
        super(BlobsDict, self).__init__(*args, **kwargs)

    def __getitem__(self, k):
        val = super(BlobsDict, self).__getitem__(k)
        if isinstance(val, dict):
            return BlobsDict(val)
        # We need to ask isinstance without actually importing blob here
        # so we accept that in order to appreciate beauty in life we must have a dash of ugliness.
        # ans instead of -
        # elif isinstance(val, Blob):
        # we ask:
        elif hasattr(val, '__class__') and val.__class__.__name__ == 'Blob':
            return val.data
        else:
            return val


class NestedBlobsDict(BlobsDict):
    """A dictionary that applies an arbitrary key-altering function
       before accessing the keys."""

    def __init__(self, *args, **kwargs):
        super(NestedBlobsDict, self).__init__(*args, **kwargs)

    def __getitem__(self, keys_str=''):

        if keys_str == '':
            return super(NestedBlobsDict, self).__getitem__(self)

        keylist = keys_str.split('.')

        cur = super(NestedBlobsDict, self).__getitem__(keylist[0])
        if len(keylist) == 1:
            return cur
        else:
            return NestedBlobsDict(cur)['.'.join(keylist[1:])]

    def __contains__(self, keys_str):
        keylist = self.keys()
        return keys_str in keylist

    def as_dict(self):
        return dict(self)

    def get(self, keys_str, default=None):
        # noinspection PyBroadException
        try:
            return self[keys_str]
        except Exception:
            return None

    def _keys(self, cur_dict, path):
        deep_keys = []
        cur_keys = dict.keys(cur_dict)

        for key in cur_keys:
            if isinstance(cur_dict[key], dict):
                if len(path) > 0:
                    deep_keys.extend(self._keys(cur_dict[key], path + '.' + key))
                else:
                    deep_keys.extend(self._keys(cur_dict[key], key))
            else:
                if len(path) > 0:
                    deep_keys.append(path + '.' + key)
                else:
                    deep_keys.append(key)

        return deep_keys

    def keys(self):
        return self._keys(self, '')


def merge_dicts(dict1, dict2):
    """ Recursively merges dict2 into dict1 """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = merge_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


def hocon_quote_key(a_dict):
    """ Recursively quote key with '.' to \"key\" """
    if not isinstance(a_dict, dict):
        return a_dict
    # preserve dict type
    new_dict = type(a_dict)()
    for k, v in a_dict.items():
        if isinstance(k, str) and '.' in k:
            new_dict['"{}"'.format(k)] = hocon_quote_key(v)
        else:
            new_dict[k] = hocon_quote_key(v)
    return new_dict


def hocon_unquote_key(a_dict):
    """ Recursively unquote \"key\" with '.' to key """
    if not isinstance(a_dict, dict):
        return a_dict
    # preserve dict type
    new_dict = type(a_dict)()
    for k, v in a_dict.items():
        if isinstance(k, str) and k[0] == '"' and k[-1] == '"' and '.' in k:
            new_dict[k[1:-1]] = hocon_unquote_key(v)
        else:
            new_dict[k] = hocon_unquote_key(v)
    return new_dict
