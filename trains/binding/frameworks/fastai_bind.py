import sys

import numpy as np

from . import _patched_call
from .tensorflow_bind import WeightsGradientHistHelper
from ..import_bind import PostImportHookPatching
from ...debugging.log import LoggerRoot


class PatchFastai(object):
    __metrics_names = None  # TODO: STORE ON OBJECT OR IN LOOKUP BASED ON OBJECT ID
    __main_task = None

    @staticmethod
    def update_current_task(task, **_):
        PatchFastai.__main_task = task
        PatchFastai._patch_model_callback()
        PostImportHookPatching.add_on_import("fastai", PatchFastai._patch_model_callback)

    @staticmethod
    def _patch_model_callback():
        # if you have tensroboard, we assume you use TesnorboardLogger, which we catch, so no need to patch.
        if "tensorboard" in sys.modules:
            return

        if "fastai" in sys.modules:
            try:
                from fastai.basic_train import Recorder

                Recorder.on_batch_end = _patched_call(Recorder.on_batch_end, PatchFastai._on_batch_end)
                Recorder.on_backward_end = _patched_call(Recorder.on_backward_end, PatchFastai._on_backward_end)
                Recorder.on_epoch_end = _patched_call(Recorder.on_epoch_end, PatchFastai._on_epoch_end)
                Recorder.on_train_begin = _patched_call(Recorder.on_train_begin, PatchFastai._on_train_begin)
            except ImportError:
                pass
            except Exception as ex:
                LoggerRoot.get_base_logger(PatchFastai).debug(str(ex))

    @staticmethod
    def _on_train_begin(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastai.__main_task:
            return
        # noinspection PyBroadException
        try:
            PatchFastai.__metrics_names = ["train_loss"] if recorder.no_val else ["train_loss", "valid_loss"]
            PatchFastai.__metrics_names += recorder.metrics_names
        except Exception:
            pass

    @staticmethod
    def _on_backward_end(original_fn, recorder, *args, **kwargs):
        def count_zeros(gradient):
            n = gradient.data.data.cpu().numpy()
            return n.size - n.count_nonzero()

        original_fn(recorder, *args, **kwargs)

        if not PatchFastai.__main_task:
            return

        # noinspection PyBroadException
        try:
            gradients = [
                x.grad.clone().detach().cpu() for x in recorder.learn.model.parameters() if x.grad is not None
            ]
            if len(gradients) == 0:
                return

            # TODO: Check computation!
            gradient_stats = np.array([
                (x.data.norm(), count_zeros(x), x.data.mean(), x.data.median(), x.data.max(), x.data.min())
                for x in gradients])
            stats_report = dict(
                avg_norm=np.mean(gradient_stats[:, 0]),
                median_norm=np.median(gradient_stats[:, 0]),
                max_norm=np.max(gradient_stats[:, 0]),
                min_norm=np.min(gradient_stats[:, 0]),
                num_zeros=gradient_stats[:, 1].sum(),
                avg_gradient=gradient_stats[:, 2].mean(),
                median_gradient=gradient_stats[:, 3].median(),
                max_gradient=gradient_stats[:, 4].max(),
                min_gradient=gradient_stats[:, 5].min(),
            )

            logger = PatchFastai.__main_task.get_logger()
            iteration = kwargs.get("iteration", 0)
            for name, val in stats_report.items():
                logger.report_scalar(title="model_stats_gradients", series=name, value=val, iteration=iteration)
        except Exception:
            pass

    @staticmethod
    def _on_epoch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastai.__main_task:
            return

        # noinspection PyBroadException
        try:
            logger = PatchFastai.__main_task.get_logger()
            iteration = kwargs.get("iteration")
            for series, value in zip(
                    PatchFastai.__metrics_names,
                    [kwargs.get("smooth_loss")] + kwargs.get("last_metrics", []),
            ):
                logger.report_scalar(title="metrics", series=series, value=value, iteration=iteration)
            PatchFastai.__main_task.flush()
        except Exception:
            pass

    @staticmethod
    def _on_batch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastai.__main_task:
            return

        # noinspection PyBroadException
        try:
            if kwargs.get("iteration") == 0 or not kwargs.get("train"):
                return

            logger = PatchFastai.__main_task.get_logger()
            logger.report_scalar(
                title="metrics",
                series="train_loss",
                value=kwargs.get("last_loss", 0),
                iteration=kwargs.get("iteration", 0)
            )
            gradient_hist_helper = WeightsGradientHistHelper(logger)
            iteration = kwargs.get("iteration")
            params = [
                (name, values.clone().detach().cpu())
                for (name, values) in recorder.model.named_parameters()
            ]
            for (name, values) in params:
                gradient_hist_helper.add_histogram(
                    title="model_weights",
                    series="model_weights/" + name,
                    step=iteration,
                    hist_data=values,
                )
        except Exception:
            pass
