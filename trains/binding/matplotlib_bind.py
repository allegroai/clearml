# -*- coding: utf-8 -*-

import os
import sys
import threading
from copy import deepcopy
from tempfile import mkstemp

import six
from six import BytesIO

from .import_bind import PostImportHookPatching
from ..config import running_remotely
from ..debugging.log import LoggerRoot
from ..utilities.resource_monitor import ResourceMonitor


class PatchedMatplotlib:
    _patched_original_plot = None
    _patched_original_figure = None
    _patched_original_savefig = None
    __patched_original_imshow = None
    __patched_original_draw_all = None
    __patched_draw_all_recursion_guard = False
    _global_plot_counter = -1
    _global_image_counter = -1
    _global_image_counter_limit = None
    _last_iteration_plot_titles = (-1, [])
    _current_task = None
    _support_image_plot = False
    _matplotlylib = None
    _plotly_renderer = None
    _lock_renderer = threading.RLock()
    _recursion_guard = {}
    _matplot_major_version = 2
    _logger_started_reporting = False
    _matplotlib_reported_titles = set()

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
        # make sure we only patch once
        PatchedMatplotlib._patched_original_plot = False

        # noinspection PyBroadException
        try:
            # we support matplotlib version 2.0.0 and above
            import matplotlib
            PatchedMatplotlib._matplot_major_version = int(matplotlib.__version__.split('.')[0])
            if PatchedMatplotlib._matplot_major_version < 2:
                LoggerRoot.get_base_logger().warning(
                    'matplotlib binding supports version 2.0 and above, found version {}'.format(
                        matplotlib.__version__))
                PatchedMatplotlib._patched_original_plot = False
                return False

            if running_remotely():
                # disable GUI backend - make headless
                matplotlib.rcParams['backend'] = 'agg'
                import matplotlib.pyplot
                matplotlib.pyplot.switch_backend('agg')
            import matplotlib.pyplot as plt
            import matplotlib.figure as figure
            if six.PY2:
                PatchedMatplotlib._patched_original_plot = staticmethod(plt.show)
                PatchedMatplotlib._patched_original_imshow = staticmethod(plt.imshow)
                PatchedMatplotlib._patched_original_figure = staticmethod(figure.Figure.show)
                PatchedMatplotlib._patched_original_savefig = staticmethod(figure.Figure.savefig)
            else:
                PatchedMatplotlib._patched_original_plot = plt.show
                PatchedMatplotlib._patched_original_imshow = plt.imshow
                PatchedMatplotlib._patched_original_figure = figure.Figure.show
                PatchedMatplotlib._patched_original_savefig = figure.Figure.savefig

            try:
                import matplotlib.pylab as pltlab
                if plt.show == pltlab.show:
                    pltlab.show = PatchedMatplotlib.patched_show
                if plt.imshow == pltlab.imshow:
                    pltlab.imshow = PatchedMatplotlib.patched_imshow
            except:
                pass
            plt.show = PatchedMatplotlib.patched_show
            figure.Figure.show = PatchedMatplotlib.patched_figure_show
            sys.modules['matplotlib'].pyplot.imshow = PatchedMatplotlib.patched_imshow
            sys.modules['matplotlib'].figure.Figure.savefig = PatchedMatplotlib.patched_savefig
            # patch plotly so we know it failed us.
            from plotly.matplotlylib import renderer
            renderer.warnings = PatchedMatplotlib._PatchWarnings()

            # ignore deprecation warnings from plotly to matplotlib
            try:
                import warnings
                warnings.filterwarnings(action='ignore', category=matplotlib.MatplotlibDeprecationWarning,
                                        module='plotly')
                warnings.filterwarnings(action='ignore', category=UserWarning, module='plotly')
            except Exception:
                pass

        except Exception:
            return False

        # patch IPython matplotlib inline mode
        # noinspection PyBroadException
        try:
            if 'IPython' in sys.modules:
                from IPython import get_ipython
                ip = get_ipython()
                if ip and matplotlib.is_interactive():
                    # instead of hooking ipython, we should hook the matplotlib
                    import matplotlib.pyplot as plt
                    PatchedMatplotlib.__patched_original_draw_all = plt.draw_all
                    plt.draw_all = PatchedMatplotlib.__patched_draw_all
                    # ip.events.register('post_execute', PatchedMatplotlib.ipython_post_execute_hook)
        except Exception:
            pass

        # update api version
        from ..backend_api import Session
        PatchedMatplotlib._support_image_plot = Session.check_min_api_version('2.2')

        # create plotly renderer
        try:
            from plotly import optional_imports
            PatchedMatplotlib._matplotlylib = optional_imports.get_module('plotly.matplotlylib')
            PatchedMatplotlib._plotly_renderer = PatchedMatplotlib._matplotlylib.PlotlyRenderer()
        except Exception:
            pass

        return True

    @staticmethod
    def update_current_task(task):
        # make sure we have a default vale
        if PatchedMatplotlib._global_image_counter_limit is None:
            from ..config import config
            PatchedMatplotlib._global_image_counter_limit = config.get('metric.matplotlib_untitled_history_size', 100)

        # if we already patched it, just update the current task
        if PatchedMatplotlib._patched_original_plot is not None:
            PatchedMatplotlib._current_task = task
        # if matplotlib is not loaded yet, get a callback hook
        elif not running_remotely() and \
                ('matplotlib.pyplot' not in sys.modules and 'matplotlib.pylab' not in sys.modules):
            PatchedMatplotlib._current_task = task
            PostImportHookPatching.add_on_import('matplotlib.pyplot', PatchedMatplotlib.patch_matplotlib)
            PostImportHookPatching.add_on_import('matplotlib.pylab', PatchedMatplotlib.patch_matplotlib)
        elif PatchedMatplotlib.patch_matplotlib():
            PatchedMatplotlib._current_task = task

    @staticmethod
    def patched_imshow(*args, **kw):
        ret = PatchedMatplotlib._patched_original_imshow(*args, **kw)
        try:
            from matplotlib import _pylab_helpers
            # store on the plot that this is an imshow plot
            stored_figure = _pylab_helpers.Gcf.get_active()
            if stored_figure:
                stored_figure._trains_is_imshow = 1 if not hasattr(stored_figure, '_trains_is_imshow') \
                    else stored_figure._trains_is_imshow + 1
        except Exception:
            pass
        return ret

    @staticmethod
    def patched_savefig(self, *args, **kw):
        ret = PatchedMatplotlib._patched_original_savefig(self, *args, **kw)
        tid = threading._get_ident() if six.PY2 else threading.get_ident()
        if not PatchedMatplotlib._recursion_guard.get(tid):
            PatchedMatplotlib._recursion_guard[tid] = True
            # noinspection PyBroadException
            try:
                PatchedMatplotlib._report_figure(specific_fig=self, set_active=False)
            except Exception:
                pass
            PatchedMatplotlib._recursion_guard[tid] = False

        return ret

    @staticmethod
    def patched_figure_show(self, *args, **kw):
        tid = threading._get_ident() if six.PY2 else threading.get_ident()
        if PatchedMatplotlib._recursion_guard.get(tid):
            # we are inside a gaurd do nothing
            return PatchedMatplotlib._patched_original_figure(self, *args, **kw)

        PatchedMatplotlib._recursion_guard[tid] = True
        PatchedMatplotlib._report_figure(set_active=False, specific_fig=self)
        ret = PatchedMatplotlib._patched_original_figure(self, *args, **kw)
        PatchedMatplotlib._recursion_guard[tid] = False
        return ret

    @staticmethod
    def patched_show(*args, **kw):
        tid = threading._get_ident() if six.PY2 else threading.get_ident()
        PatchedMatplotlib._recursion_guard[tid] = True
        # noinspection PyBroadException
        try:
            figures = PatchedMatplotlib._get_output_figures(None, all_figures=True)
            for figure in figures:
                # if this is a stale figure (just updated) we should send it, the rest will not be stale
                if figure.canvas.figure.stale or (hasattr(figure, '_trains_is_imshow') and figure._trains_is_imshow):
                    PatchedMatplotlib._report_figure(stored_figure=figure)
        except Exception:
            pass
        ret = PatchedMatplotlib._patched_original_plot(*args, **kw)
        if PatchedMatplotlib._current_task and sys.modules['matplotlib'].rcParams['backend'] == 'agg':
            # clear the current plot, because no one else will
            # noinspection PyBroadException
            try:
                if sys.modules['matplotlib'].rcParams['backend'] == 'agg':
                    import matplotlib.pyplot as plt
                    plt.clf()
            except Exception:
                pass
        PatchedMatplotlib._recursion_guard[tid] = False
        return ret

    @staticmethod
    def _report_figure(force_save_as_image=False, stored_figure=None, set_active=True, specific_fig=None):
        if not PatchedMatplotlib._current_task:
            return

        # noinspection PyBroadException
        try:
            import matplotlib.pyplot as plt
            from matplotlib import _pylab_helpers
            from plotly.io import templates
            if specific_fig is None:
                # store the figure object we just created (if it is not already there)
                stored_figure = stored_figure or _pylab_helpers.Gcf.get_active()
                if not stored_figure:
                    # nothing for us to do
                    return
                # check if this is an imshow
                if hasattr(stored_figure, '_trains_is_imshow'):
                    # flag will be cleared when calling clf() (object will be replaced)
                    stored_figure._trains_is_imshow = max(0, stored_figure._trains_is_imshow - 1)
                    force_save_as_image = True
                # get current figure
                mpl_fig = stored_figure.canvas.figure  # plt.gcf()
            else:
                mpl_fig = specific_fig

            # convert to plotly
            image = None
            plotly_fig = None
            image_format = 'jpeg'
            fig_dpi = 300
            if force_save_as_image:
                # if this is an image, store as is.
                fig_dpi = None
            else:
                image_format = 'svg'
                # protect with lock, so we support multiple threads using the same renderer
                PatchedMatplotlib._lock_renderer.acquire()
                # noinspection PyBroadException
                try:
                    def our_mpl_to_plotly(fig):
                        if not PatchedMatplotlib._matplotlylib or not PatchedMatplotlib._plotly_renderer:
                            return None
                        PatchedMatplotlib._matplotlylib.Exporter(PatchedMatplotlib._plotly_renderer,
                                                                 close_mpl=False).run(fig)
                        x_ticks = list(PatchedMatplotlib._plotly_renderer.current_mpl_ax.get_xticklabels())
                        if x_ticks:
                            try:
                                # check if all values can be cast to float
                                values = [float(t.get_text().replace('−', '-')) for t in x_ticks]
                            except:
                                try:
                                    PatchedMatplotlib._plotly_renderer.plotly_fig['layout']['xaxis1'].update({
                                        'ticktext': [t.get_text() for t in x_ticks],
                                        'tickvals': [t.get_position()[0] for t in x_ticks],
                                    })
                                except:
                                    pass
                        y_ticks = list(PatchedMatplotlib._plotly_renderer.current_mpl_ax.get_yticklabels())
                        if y_ticks:
                            try:
                                # check if all values can be cast to float
                                values = [float(t.get_text().replace('−', '-')) for t in y_ticks]
                            except:
                                try:
                                    PatchedMatplotlib._plotly_renderer.plotly_fig['layout']['yaxis1'].update({
                                        'ticktext': [t.get_text() for t in y_ticks],
                                        'tickvals': [t.get_position()[1] for t in y_ticks],
                                    })
                                except:
                                    pass
                        return deepcopy(PatchedMatplotlib._plotly_renderer.plotly_fig)

                    plotly_fig = our_mpl_to_plotly(mpl_fig)
                    try:
                        if 'none' in templates:
                            plotly_fig._layout_obj.template = templates['none']
                    except:
                        pass
                except Exception as ex:
                    # this was an image, change format to png
                    image_format = 'jpeg' if 'selfie' in str(ex) else 'png'
                    fig_dpi = 300
                finally:
                    PatchedMatplotlib._lock_renderer.release()

            # plotly could not serialize the plot, we should convert to image
            if not plotly_fig:
                plotly_fig = None
                # noinspection PyBroadException
                try:
                    # first try SVG if we fail then fallback to png
                    buffer_ = BytesIO()
                    a_plt = specific_fig if specific_fig is not None else plt
                    if PatchedMatplotlib._matplot_major_version < 3:
                        a_plt.savefig(buffer_, dpi=fig_dpi, format=image_format, bbox_inches='tight', pad_inches=0,
                                      frameon=False)
                    else:
                        a_plt.savefig(buffer_, dpi=fig_dpi, format=image_format, bbox_inches='tight', pad_inches=0,
                                      facecolor=None)
                    buffer_.seek(0)
                except Exception:
                    image_format = 'png'
                    buffer_ = BytesIO()
                    a_plt = specific_fig if specific_fig is not None else plt
                    if PatchedMatplotlib._matplot_major_version < 3:
                        a_plt.savefig(buffer_, dpi=fig_dpi, format=image_format, bbox_inches='tight', pad_inches=0,
                                      frameon=False)
                    else:
                        a_plt.savefig(buffer_, dpi=fig_dpi, format=image_format, bbox_inches='tight', pad_inches=0,
                                      facecolor=None)
                    buffer_.seek(0)
                fd, image = mkstemp(suffix='.' + image_format)
                os.write(fd, buffer_.read())
                os.close(fd)

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
                    last_iteration = PatchedMatplotlib._get_last_iteration()
                    if plot_title:
                        title = PatchedMatplotlib._enforce_unique_title_per_iteration(plot_title, last_iteration)
                    else:
                        PatchedMatplotlib._global_plot_counter += 1
                        title = 'untitled %d' % PatchedMatplotlib._global_plot_counter

                    plotly_fig.layout.margin = {}
                    plotly_fig.layout.autosize = True
                    plotly_fig.layout.height = None
                    plotly_fig.layout.width = None
                    # send the plot event
                    plotly_dict = plotly_fig.to_plotly_json()
                    if not plotly_dict.get('layout'):
                        plotly_dict['layout'] = {}
                    plotly_dict['layout']['title'] = title

                    PatchedMatplotlib._matplotlib_reported_titles.add(title)
                    reporter.report_plot(title=title, series='plot', plot=plotly_dict, iter=last_iteration)
                else:
                    logger = PatchedMatplotlib._current_task.get_logger()

                    # this is actually a failed plot, we should put it under plots:
                    # currently disabled
                    if force_save_as_image or not PatchedMatplotlib._support_image_plot:
                        last_iteration = PatchedMatplotlib._get_last_iteration()
                        # send the plot as image
                        if plot_title:
                            title = PatchedMatplotlib._enforce_unique_title_per_iteration(plot_title, last_iteration)
                        else:
                            PatchedMatplotlib._global_image_counter += 1
                            title = 'untitled %d' % (PatchedMatplotlib._global_image_counter %
                                                     PatchedMatplotlib._global_image_counter_limit)

                        PatchedMatplotlib._matplotlib_reported_titles.add(title)
                        logger.report_image(title=title, series='plot image', local_path=image,
                                            delete_after_upload=True, iteration=last_iteration)
                    else:
                        # send the plot as plotly with embedded image
                        last_iteration = PatchedMatplotlib._get_last_iteration()
                        if plot_title:
                            title = PatchedMatplotlib._enforce_unique_title_per_iteration(plot_title, last_iteration)
                        else:
                            PatchedMatplotlib._global_plot_counter += 1
                            title = 'untitled %d' % (PatchedMatplotlib._global_plot_counter %
                                                     PatchedMatplotlib._global_image_counter_limit)

                        PatchedMatplotlib._matplotlib_reported_titles.add(title)
                        logger._report_image_plot_and_upload(title=title, series='plot image', path=image,
                                                             delete_after_upload=True, iteration=last_iteration)
        except Exception:
            # plotly failed
            pass

        return

    @staticmethod
    def _enforce_unique_title_per_iteration(title, last_iteration):
        if last_iteration != PatchedMatplotlib._last_iteration_plot_titles[0]:
            PatchedMatplotlib._last_iteration_plot_titles = (last_iteration, [title])
        elif title not in PatchedMatplotlib._last_iteration_plot_titles[1]:
            PatchedMatplotlib._last_iteration_plot_titles[1].append(title)
        else:
            base_title = title
            counter = 1
            while title in PatchedMatplotlib._last_iteration_plot_titles[1]:
                # we already used this title in this iteration, we should change the title
                title = base_title + ' %d' % counter
                counter += 1
            # store the new title
            PatchedMatplotlib._last_iteration_plot_titles[1].append(title)
        return title

    @staticmethod
    def _get_output_figures(stored_figure, all_figures):
        try:
            from matplotlib import _pylab_helpers
            if all_figures:
                return list(_pylab_helpers.Gcf.figs.values())
            else:
                return [stored_figure] or [_pylab_helpers.Gcf.get_active()]
        except Exception:
            return []

    @staticmethod
    def __patched_draw_all(*args, **kwargs):
        recursion_guard = PatchedMatplotlib.__patched_draw_all_recursion_guard
        if not recursion_guard:
            PatchedMatplotlib.__patched_draw_all_recursion_guard = True

        ret = PatchedMatplotlib.__patched_original_draw_all(*args, **kwargs)

        if not recursion_guard:
            PatchedMatplotlib.ipython_post_execute_hook()
            PatchedMatplotlib.__patched_draw_all_recursion_guard = False
        return ret

    @staticmethod
    def _get_last_iteration():
        if PatchedMatplotlib._logger_started_reporting:
            return PatchedMatplotlib._current_task.get_last_iteration()
        # get the reported plot titles (exclude us)
        reported_titles = ResourceMonitor.get_logger_reported_titles(PatchedMatplotlib._current_task)
        if not reported_titles:
            return 0
        # check that this is not only us
        if not (set(reported_titles) - PatchedMatplotlib._matplotlib_reported_titles):
            return 0
        # mark reporting started
        PatchedMatplotlib._logger_started_reporting = True
        return PatchedMatplotlib._current_task.get_last_iteration()

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
