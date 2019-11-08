

class ProxyDictPostWrite(dict):
    """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """

    def __init__(self, update_obj, update_func, *args, **kwargs):
        super(ProxyDictPostWrite, self).__init__(*args, **kwargs)
        self._update_func = None
        for k, i in self.items():
            if isinstance(i, dict):
                self.update({k: ProxyDictPostWrite(update_obj, self._set_callback, **i)})
        self._update_obj = update_obj
        self._update_func = update_func

    def __setitem__(self, key, value):
        super(ProxyDictPostWrite, self).__setitem__(key, value)
        self._set_callback()

    def _set_callback(self, *_):
        if self._update_func:
            self._update_func(self._update_obj, self)


class ProxyDictPreWrite(dict):
    """ Dictionary wrapper that prevents modifications to the dictionary """

    def __init__(self, update_obj, update_func, *args, **kwargs):
        super(ProxyDictPreWrite, self).__init__(*args, **kwargs)
        self._update_func = None
        for k, i in self.items():
            if isinstance(i, dict):
                self.update({k: ProxyDictPreWrite(k, self._nested_callback, **i)})
        self._update_obj = update_obj
        self._update_func = update_func

    def __setitem__(self, key, value):
        key_value = self._set_callback((key, value,))
        if key_value:
            super(ProxyDictPreWrite, self).__setitem__(*key_value)

    def _set_callback(self, key_value, *_):
        if self._update_func:
            res = self._update_func(self._update_obj, key_value)
            if not res:
                return None
            return res
        return key_value

    def _nested_callback(self, prefix, key_value):
        return self._set_callback((prefix+'.'+key_value[0], key_value[1],))
