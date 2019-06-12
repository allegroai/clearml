import sys

import cv2
import numpy as np
import six
from six import BytesIO

from ..debugging.log import LoggerRoot
from ..config import running_remotely


class PatchedMatplotlib:
    _patched_original_plot = None
    __patched_original_imshow = None
    _global_plot_counter = -1
    _global_image_counter = -1
    _current_task = None

    class _PatchWarnings(object):
        def __init__(self):
            pass

        def warn(self, text, *args, **kwargs):
            raise ValueError(text)

        def __getattr__(self, item):
            def bypass(*args, **kwargs):
                pass
            return bypass

    @staticmethod
    def patch_matplotlib():
        # only once
        if PatchedMatplotlib._patched_original_plot is not None:
            return True
        # noinspection PyBroadException
        try:
            # we support matplotlib version 2.0.0 and above
            import matplotlib
            if int(matplotlib.__version__.split('.')[0]) < 2:
                LoggerRoot.get_base_logger().warning(
                    'matplotlib binding supports version 2.0 and above, found version {}'.format(
                        matplotlib.__version__))
                return False

            if running_remotely():
                # disable GUI backend - make headless
                sys.modules['matplotlib'].rcParams['backend'] = 'agg'
                import matplotlib.pyplot
                sys.modules['matplotlib'].pyplot.switch_backend('agg')
            import matplotlib.pyplot as plt
            from matplotlib import _pylab_helpers
            if six.PY2:
                PatchedMatplotlib._patched_original_plot = staticmethod(sys.modules['matplotlib'].pyplot.show)
                PatchedMatplotlib._patched_original_imshow = staticmethod(sys.modules['matplotlib'].pyplot.imshow)
            else:
                PatchedMatplotlib._patched_original_plot = sys.modules['matplotlib'].pyplot.show
                PatchedMatplotlib._patched_original_imshow = sys.modules['matplotlib'].pyplot.imshow
            sys.modules['matplotlib'].pyplot.show = PatchedMatplotlib.patched_show
            # sys.modules['matplotlib'].pyplot.imshow = PatchedMatplotlib.patched_imshow
            # patch plotly so we know it failed us.
            from plotly.matplotlylib import renderer
            renderer.warnings = PatchedMatplotlib._PatchWarnings()
        except Exception:
            return False

        # patch IPython matplotlib inline mode
        # noinspection PyBroadException
        try:
            if 'IPython' in sys.modules:
                from IPython import get_ipython
                ip = get_ipython()
                if ip and matplotlib.is_interactive():
                    ip.events.register('post_execute', PatchedMatplotlib.ipython_post_execute_hook)
        except Exception:
            pass

        return True

    @staticmethod
    def update_current_task(task):
        if PatchedMatplotlib.patch_matplotlib():
            PatchedMatplotlib._current_task = task

    @staticmethod
    def patched_imshow(*args, **kw):
        ret = PatchedMatplotlib._patched_original_imshow(*args, **kw)
        PatchedMatplotlib._report_figure(force_save_as_image=True)
        return ret

    @staticmethod
    def patched_show(*args, **kw):
        PatchedMatplotlib._report_figure()
        ret = PatchedMatplotlib._patched_original_plot(*args, **kw)
        if PatchedMatplotlib._current_task and running_remotely():
            # clear the current plot, because no one else will
            # noinspection PyBroadException
            try:
                if sys.modules['matplotlib'].rcParams['backend'] == 'agg':
                    import matplotlib.pyplot as plt
                    plt.clf()
            except Exception:
                pass
        return ret

    @staticmethod
    def _report_figure(force_save_as_image=False, stored_figure=None, set_active=True):
        if not PatchedMatplotlib._current_task:
            return

        # noinspection PyBroadException
        try:
            import matplotlib.pyplot as plt
            from plotly import optional_imports
            from matplotlib import _pylab_helpers
            # store the figure object we just created (if it is not already there)
            stored_figure = stored_figure or _pylab_helpers.Gcf.get_active()
            if not stored_figure:
                # nothing for us to do
                return
            # get current figure
            mpl_fig = stored_figure.canvas.figure  # plt.gcf()
            # convert to plotly
            image = None
            plotly_fig = None
            if not force_save_as_image:
                # noinspection PyBroadException
                try:
                    def our_mpl_to_plotly(fig):
                        matplotlylib = optional_imports.get_module('plotly.matplotlylib')
                        if matplotlylib:
                            renderer = matplotlylib.PlotlyRenderer()
                            matplotlylib.Exporter(renderer, close_mpl=False).run(fig)
                            return renderer.plotly_fig

                    plotly_fig = our_mpl_to_plotly(mpl_fig)
                except Exception:
                    pass

            # plotly could not serialize the plot, we should convert to image
            if not plotly_fig:
                plotly_fig = None
                buffer_ = BytesIO()
                plt.savefig(buffer_, format="png", bbox_inches='tight', pad_inches=0)
                buffer_.seek(0)
                buffer = buffer_.getbuffer() if not six.PY2 else buffer_.getvalue()
                image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # check if we need to restore the active object
            if set_active and not _pylab_helpers.Gcf.get_active():
                _pylab_helpers.Gcf.set_active(stored_figure)

            # get the main task
            reporter = PatchedMatplotlib._current_task.reporter
            if reporter is not None:
                if mpl_fig.texts:
                    plot_title = mpl_fig.texts[0].get_text()
                else:
                    gca = mpl_fig.gca()
                    plot_title = gca.title.get_text() if gca.title else None

                # remove borders and size, we should let the web take care of that
                if plotly_fig:
                    PatchedMatplotlib._global_plot_counter += 1
                    title = plot_title or 'untitled %d' % PatchedMatplotlib._global_plot_counter
                    plotly_fig.layout.margin = {}
                    plotly_fig.layout.autosize = True
                    plotly_fig.layout.height = None
                    plotly_fig.layout.width = None
                    # send the plot event
                    reporter.report_plot(title=title, series='plot', plot=plotly_fig.to_plotly_json(),
                                         iter=PatchedMatplotlib._global_plot_counter if plot_title else 0)
                else:
                    # send the plot as image
                    PatchedMatplotlib._global_image_counter += 1
                    logger = PatchedMatplotlib._current_task.get_logger()
                    title = plot_title or 'untitled %d' % PatchedMatplotlib._global_image_counter
                    logger.report_image_and_upload(title=title, series='plot image', matrix=image,
                                                   iteration=PatchedMatplotlib._global_image_counter
                                                   if plot_title else 0)
        except Exception:
            # plotly failed
            pass

        return

    @staticmethod
    def ipython_post_execute_hook():
        # noinspection PyBroadException
        try:
            from matplotlib import _pylab_helpers
            for i, f_mgr in enumerate(_pylab_helpers.Gcf.get_all_fig_managers()):
                if not f_mgr.canvas.figure.stale:
                    PatchedMatplotlib._report_figure(stored_figure=f_mgr)
        except Exception:
            pass
