import attr

from .version import Version


class attrs(object):
    def __init__(self, *args, **kwargs):
        if any(x in kwargs for x in ("eq", "order")):
            raise RuntimeError("Only `cmp` is supported for attr.attrs, not `eq` or `order`")
        if Version(attr.__version__) >= Version("19.2"):
            cmp = kwargs.pop("cmp", None)
            if cmp is not None:
                kwargs["eq"] = kwargs["order"] = cmp
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        return attr.attrs(*self.args, **self.kwargs)(f)
