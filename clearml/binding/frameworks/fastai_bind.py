import sys

import numpy as np

from clearml.utilities.version import Version

from . import _patched_call
from .tensorflow_bind import WeightsGradientHistHelper
from ..import_bind import PostImportHookPatching
from ...debugging.log import LoggerRoot

try:
    import fastai
except ImportError:
    fastai = None


class PatchFastai(object):
    @staticmethod
    def update_current_task(task, **_):
        if fastai is None:
            return

        # noinspection PyBroadException
        try:
            if Version(fastai.__version__) < Version("2.0.0"):
                PatchFastaiV1.update_current_task(task)
                PostImportHookPatching.add_on_import("fastai", PatchFastaiV1.patch_model_callback)
            else:
                PatchFastaiV2.update_current_task(task)
                PostImportHookPatching.add_on_import("fastai", PatchFastaiV2.patch_model_callback)
        except Exception:
            pass


class PatchFastaiV1(object):
    __metrics_names = {}
    __gradient_hist_helpers = {}
    _current_task = None
    __patched = False

    @staticmethod
    def update_current_task(task, **_):
        PatchFastaiV1._current_task = task
        if not task:
            return
        if not PatchFastaiV1.__patched:
            PatchFastaiV1.__patched = True
            PatchFastaiV1.patch_model_callback()

    @staticmethod
    def patch_model_callback():
        # if you have tensorboard, we assume you use TensorboardLogger, which we catch, so no need to patch.
        if "tensorboard" in sys.modules:
            return

        try:
            from fastai.basic_train import Recorder
            Recorder.on_batch_end = _patched_call(Recorder.on_batch_end, PatchFastaiV1._on_batch_end)
            Recorder.on_backward_end = _patched_call(Recorder.on_backward_end, PatchFastaiV1._on_backward_end)
            Recorder.on_epoch_end = _patched_call(Recorder.on_epoch_end, PatchFastaiV1._on_epoch_end)
            Recorder.on_train_begin = _patched_call(Recorder.on_train_begin, PatchFastaiV1._on_train_begin)
        except ImportError:
            pass
        except Exception as ex:
            LoggerRoot.get_base_logger(PatchFastaiV1).debug(str(ex))

    @staticmethod
    def _on_train_begin(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastaiV1._current_task:
            return
        # noinspection PyBroadException
        try:
            PatchFastaiV1.__metrics_names[id(recorder)] = (
                ["train_loss"] if recorder.no_val else ["train_loss", "valid_loss"]
            )
            PatchFastaiV1.__metrics_names[id(recorder)] += recorder.metrics_names
        except Exception:
            pass

    @staticmethod
    def _on_backward_end(original_fn, recorder, *args, **kwargs):
        def count_zeros(gradient):
            n = gradient.data.data.cpu().numpy()
            return n.size - np.count_nonzero(n)

        original_fn(recorder, *args, **kwargs)

        if not PatchFastaiV1._current_task:
            return

        # noinspection PyBroadException
        try:
            gradients = [x.grad.clone().detach().cpu() for x in recorder.learn.model.parameters() if x.grad is not None]
            if len(gradients) == 0:
                return

            # TODO: Check computation!
            gradient_stats = np.array(
                [
                    (x.data.norm(), count_zeros(x), x.data.mean(), np.median(x.data), x.data.max(), x.data.min())
                    for x in gradients
                ]
            )
            stats_report = dict(
                avg_norm=np.mean(gradient_stats[:, 0]),
                median_norm=np.median(gradient_stats[:, 0]),
                max_norm=np.max(gradient_stats[:, 0]),
                min_norm=np.min(gradient_stats[:, 0]),
                num_zeros=gradient_stats[:, 1].sum(),
                avg_gradient=gradient_stats[:, 2].mean(),
                median_gradient=np.median(gradient_stats[:, 3]),
                max_gradient=gradient_stats[:, 4].max(),
                min_gradient=gradient_stats[:, 5].min(),
            )

            logger = PatchFastaiV1._current_task.get_logger()
            iteration = kwargs.get("iteration", 0)
            for name, val in stats_report.items():
                logger.report_scalar(title="model_stats_gradients", series=name, value=val, iteration=iteration)
        except Exception:
            pass

    @staticmethod
    def _on_epoch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastaiV1._current_task:
            return

        # noinspection PyBroadException
        try:
            logger = PatchFastaiV1._current_task.get_logger()
            iteration = kwargs.get("iteration")
            for series, value in zip(
                PatchFastaiV1.__metrics_names[id(recorder)],
                [kwargs.get("smooth_loss")] + kwargs.get("last_metrics", []),
            ):
                logger.report_scalar(title="metrics", series=series, value=value, iteration=iteration)
            PatchFastaiV1._current_task.flush()
        except Exception:
            pass

    @staticmethod
    def _on_batch_end(original_fn, recorder, *args, **kwargs):
        original_fn(recorder, *args, **kwargs)
        if not PatchFastaiV1._current_task:
            return

        # noinspection PyBroadException
        try:
            iteration = kwargs.get("iteration", 0)
            if iteration == 0 or not kwargs.get("train"):
                return

            logger = PatchFastaiV1._current_task.get_logger()
            logger.report_scalar(
                title="metrics",
                series="train_loss",
                value=kwargs.get("last_loss", 0),
                iteration=iteration,
            )
            params = [(name, values.clone().detach().cpu()) for (name, values) in recorder.model.named_parameters()]
            if (
                id(recorder) not in PatchFastaiV1.__gradient_hist_helpers
                or PatchFastaiV1.__gradient_hist_helpers[id(recorder)].logger is not logger
            ):
                PatchFastaiV1.__gradient_hist_helpers[id(recorder)] = WeightsGradientHistHelper(logger)
            histograms = []
            for (name, values) in params:
                histograms.append(
                    dict(title="model_weights", series="model_weights/" + name, step=iteration, hist_data=values)
                )
            PatchFastaiV1.__gradient_hist_helpers[id(recorder)].add_histograms(histograms)
        except Exception:
            pass


class PatchFastaiV2(object):
    _current_task = None
    __patched = False

    @staticmethod
    def update_current_task(task, **_):
        PatchFastaiV2._current_task = task
        if not task:
            return
        if not PatchFastaiV2.__patched:
            PatchFastaiV2.__patched = True
            PatchFastaiV2.patch_model_callback()

    @staticmethod
    def patch_model_callback():
        if "tensorboard" in sys.modules:
            return

        # noinspection PyBroadException
        try:
            fastai.learner.Learner.fit = _patched_call(fastai.learner.Learner.fit, PatchFastaiV2._insert_callbacks)
        except Exception:
            pass

    try:
        from fastai.learner import Recorder

        __patch_fastai_callbacks_base = Recorder
    except ImportError:
        __patch_fastai_callbacks_base = object

    class PatchFastaiCallbacks(__patch_fastai_callbacks_base):
        __id = 0

        def __init__(self, *args, **kwargs):
            kwargs["train_metrics"] = True
            super().__init__(*args, **kwargs)
            self.__train_iter = 0

            def noop(*_, **__):
                pass

            self.logger = noop
            self.__id = str(PatchFastaiV2.PatchFastaiCallbacks.__id)
            PatchFastaiV2.PatchFastaiCallbacks.__id += 1
            self.__gradient_hist_helper = WeightsGradientHistHelper(PatchFastaiV2._current_task.get_logger())

        def after_batch(self):
            # noinspection PyBroadException
            try:
                super().after_batch()  # noqa
                if not PatchFastaiV2._current_task:
                    return
                logger = PatchFastaiV2._current_task.get_logger()
                if not self.training:  # noqa
                    return
                self.__train_iter += 1
                for metric in self._train_mets:  # noqa
                    logger.report_scalar(
                        title="metrics_" + self.__id,
                        series="train_" + metric.name,
                        value=metric.value,
                        iteration=self.__train_iter,
                    )
                for k, v in self.opt.hypers[-1].items():  # noqa
                    logger.report_scalar(title=k + "_" + self.__id, series=k, value=v, iteration=self.__train_iter)
                params = [
                    (name, values.clone().detach().cpu()) for (name, values) in self.model.named_parameters()
                ]  # noqa
                if self.__gradient_hist_helper.logger is not logger:
                    self.__gradient_hist_helper = WeightsGradientHistHelper(logger)
                histograms = []
                for (name, values) in params:
                    histograms.append(
                        dict(
                            title="model_weights_" + self.__id,
                            series="model_weights/" + name,
                            step=self.__train_iter,
                            hist_data=values,
                        )
                    )
                self.__gradient_hist_helper.add_histograms(histograms)
            except Exception:
                pass

        def after_epoch(self):
            # noinspection PyBroadException
            try:
                super().after_epoch()  # noqa
                if not PatchFastaiV2._current_task:
                    return
                logger = PatchFastaiV2._current_task.get_logger()
                for metric in self._valid_mets:  # noqa
                    logger.report_scalar(
                        title="metrics_" + self.__id,
                        series="valid_" + metric.name,
                        value=metric.value,
                        iteration=self.__train_iter,
                    )
            except Exception:
                pass

        def before_step(self):
            # noinspection PyBroadException
            try:
                if hasattr(fastai.learner.Recorder, "before_step"):
                    super().before_step()  # noqa
                if not PatchFastaiV2._current_task:
                    return
                logger = PatchFastaiV2._current_task.get_logger()
                gradients = [
                    x.grad.clone().detach().cpu() for x in self.learn.model.parameters() if x.grad is not None
                ]  # noqa
                if len(gradients) == 0:
                    return

                def count_zeros(gradient):
                    n = gradient.data.data.cpu().numpy()
                    return n.size - np.count_nonzero(n)

                gradient_stats = np.array(
                    [
                        (x.data.norm(), count_zeros(x), x.data.mean(), np.median(x.data), x.data.max(), x.data.min())
                        for x in gradients
                    ]
                )
                # TODO: Check computation!
                stats_report = dict(
                    avg_norm=np.mean(gradient_stats[:, 0]),
                    median_norm=np.median(gradient_stats[:, 0]),
                    max_norm=np.max(gradient_stats[:, 0]),
                    min_norm=np.min(gradient_stats[:, 0]),
                    num_zeros=gradient_stats[:, 1].sum(),
                    avg_gradient=gradient_stats[:, 2].mean(),
                    median_gradient=np.median(gradient_stats[:, 3]),
                    max_gradient=gradient_stats[:, 4].max(),
                    min_gradient=gradient_stats[:, 5].min(),
                )
                for name, val in stats_report.items():
                    if name != "num_zeros":
                        title = "model_stats_gradients_" + self.__id
                    else:
                        title = "model_stats_gradients_num_zeros_" + self.__id
                    logger.report_scalar(title=title, series=name, value=val, iteration=self.__train_iter)
            except Exception:
                pass

    @staticmethod
    def _insert_callbacks(original_fn, obj, *args, **kwargs):
        obj.add_cb(PatchFastaiV2.PatchFastaiCallbacks)
        return original_fn(obj, *args, **kwargs)
