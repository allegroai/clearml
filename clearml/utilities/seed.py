import sys
import random

try:
    import numpy as np
except Exception:
    np = None


def make_deterministic(seed=1337, cudnn_deterministic=False):
    """
    Ensure deterministic behavior across PyTorch using the provided random seed.
    This function makes sure that torch, numpy and random use the same random seed.

    When using clearml's task, call this function using the task's random seed like so:
        make_deterministic(task.get_random_seed())

    :param int seed: Seed number
    :param bool cudnn_deterministic: In order to make computations deterministic on your specific platform
    and PyTorch release, set this value to True. torch will only allow those CuDNN algorithms that are
    (believed to be) deterministic. This can have a performance impact (slower execution) depending on your model.
    """
    seed = int(seed) & 0xFFFFFFFF
    torch = sys.modules.get("torch")
    tf = sys.modules.get("tensorflow")

    if cudnn_deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        except Exception:
            pass

    if tf is not None:
        # reset graph state
        try:
            import tensorflow  # noqa: F401
            from tensorflow.python.eager.context import _context
            eager_mode_bypass = _context is None
        except Exception:
            eager_mode_bypass = False

        if not eager_mode_bypass:
            try:
                tf.compat.v1.set_random_seed(seed)
            except Exception:
                try:
                    tf.set_random_seed(seed)
                except Exception:
                    pass

            try:
                tf.compat.v1.random.set_random_seed(seed)
            except Exception:
                try:
                    tf.random.set_random_seed(seed)
                except Exception:
                    pass
