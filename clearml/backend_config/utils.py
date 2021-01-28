
def get_items(cls):
    """ get key/value items from an enum-like class (members represent enumeration key/value) """
    return {k: v for k, v in vars(cls).items() if not k.startswith('_')}


def get_options(cls):
    """ get options from an enum-like class (members represent enumeration key/value) """
    return get_items(cls).values()
