import statistics
import sys

import numpy as np

from . import _patched_call
from .tensorflow_bind import WeightsGradientHistHelper
from ..import_bind import PostImportHookPatching
from ...debugging.log import LoggerRoot


class PatchFastai(object):
    __metrics_names = None
    __main_task = None

    @staticmethod
    def update_current_task(task, **kwargs):
        PatchFastai.__main_task = task
        PatchFastai._patch_model_callback()
        PostImportHookPatching.add_on_import(
            "fastai", PatchFastai._patch_model_callback
        )

    @staticmethod
    def _patch_model_callback():
        if "fastai" in sys.modules:
            try:
                from fastai.basic_train import Recorder

                Recorder.on_batch_end = _patched_call(
                    Recorder.on_batch_end, PatchFastai._on_batch_end
                )
                Recorder.on_backward_end = _patched_call(
                    Recorder.on_backward_end, PatchFastai._on_backward_end
                )
                Recorder.on_epoch_end = _patched_call(
                    Recorder.on_epoch_end, PatchFastai._on_epoch_end
                )
                Recorder.on_train_begin = _patched_call(
                    Recorder.on_train_begin, PatchFastai._on_train_begin
                )

            except ImportError:
                pass
            except Exception as ex:
                LoggerRoot.get_base_logger(PatchFastai).debug(str(ex))

    @staticmethod
    def _on_train_begin(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        PatchFastai.__metrics_names = (
            ["train_loss"] if recorder.no_val else ["train_loss", "valid_loss"]
        )
        PatchFastai.__metrics_names += recorder.metrics_names

    @staticmethod
    def _on_backward_end(original_fn, recorder, *args, **kwargs):
        def report_model_stats(series, value):
            logger.report_scalar("model_stats_gradients", series, value, iteration)

        original_fn(recorder, *args, **kwargs)
        gradients = [
            x.grad.clone().detach().cpu()
            for x in recorder.learn.model.parameters()
            if x.grad is not None
        ]
        if len(gradients) == 0:
            return
        iteration = kwargs.get("iteration")
        norms = [x.data.norm() for x in gradients]
        logger = PatchFastai.__main_task.get_logger()
        for name, val in zip(
            [
                "avg_norm",
                "median_norm",
                "max_norm",
                "min_norm",
                "num_zeros",
                "avg_gradient",
                "median_gradient",
                "max_gradient",
                "min_gradient",
            ],
            [
                sum(norms) / len(gradients),
                statistics.median(norms),
                max(norms),
                min(norms),
                sum(
                    (np.asarray(x) == 0.0).sum()
                    for x in [x.data.data.cpu().numpy() for x in gradients]
                ),
                sum(x.data.mean() for x in gradients) / len(gradients),
                statistics.median(x.data.median() for x in gradients),
                max(x.data.max() for x in gradients),
                min(x.data.min() for x in gradients),
            ],
        ):
            report_model_stats(name, val)

    @staticmethod
    def _on_epoch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        logger = PatchFastai.__main_task.get_logger()
        iteration = kwargs.get("iteration")
        for series, value in zip(
            PatchFastai.__metrics_names,
            [kwargs.get("smooth_loss")] + kwargs.get("last_metrics", []),
        ):
            logger.report_scalar("metrics", series, value, iteration)
        PatchFastai.__main_task.flush()

    @staticmethod
    def _on_batch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if kwargs.get("iteration") == 0 or not kwargs.get("train"):
            return
        logger = PatchFastai.__main_task.get_logger()
        logger.report_scalar(
            "metrics", "train_loss", kwargs.get("last_loss"), kwargs.get("iteration")
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
