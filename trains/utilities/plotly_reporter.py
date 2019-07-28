import numpy as np
from attr import attrs, attrib


def create_2d_histogram_plot(np_row_wise, labels, title=None, xtitle=None, ytitle=None, series=None, xlabels=None,
                             comment=None):
    """
    Create a 2D Plotly histogram chart from a 2D numpy array
    :param np_row_wise: 2D numpy data array
    :param labels: Histogram labels
    :param title: Chart title
    :param xtitle: X-Series title
    :param ytitle: Y-Series title
    :param comment: comment underneath the title
    :return: Plotly chart dict
    """
    np_row_wise = np.atleast_2d(np_row_wise)
    assert len(np_row_wise.shape) == 2, "Expected a 2D numpy array"
    # using labels without xlabels leads to original behavior
    if labels is not None and xlabels is None:
        assert len(labels) == np_row_wise.shape[0], "Please provide a label for each data row"
    elif xlabels is None:
        fake_label = series or ''
        labels = [fake_label] * np_row_wise.shape[0]
    elif labels:
        if len(labels) == 1:
            labels = [labels] * np_row_wise.shape[0]
        assert len(xlabels) == np_row_wise.shape[1]

    data = [_np_row_to_plotly_data_item(np_row=np_row_wise[i, :], label=labels[i] if labels else None, xlabels=xlabels)
            for i in range(np_row_wise.shape[0])]
    return _plotly_hist_dict(title=title, xtitle=xtitle, ytitle=ytitle, data=data, comment=comment)


def _to_np_array(value):
    if not isinstance(value, np.ndarray):
        value = np.array(value)

    return value


@attrs
class SeriesInfo(object):
    name = attrib(type=str)
    data = attrib(type=np.ndarray, converter=_to_np_array)
    labels = attrib(default=None)

    @data.validator
    def _validate_data(self, _, value):
        if value.ndim != 2:
            raise ValueError("Expected series data to be 2D numpy array")

        if value.shape[1] != 2:
            raise ValueError("Expected series data to have 2 columns")

    def __attrs_post_init__(self):
        if (self.labels is not None) and (len(self.labels) != self.data.shape[0]):
            raise ValueError(
                "If 'labels' is provided, it must be a list or tuple, "
                "the same length as the data"
            )


def create_line_plot(title, series, xtitle, ytitle, mode='lines', reverse_xaxis=False, comment=None):
    plotly_obj = _plotly_scatter_layout_dict(
        title=title if not comment else (title + '<br><sup>' + comment + '</sup>'),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
    )

    if reverse_xaxis:
        plotly_obj["layout"]["xaxis"]["autorange"] = "reversed"

    # check maximum size of data
    _MAX_SIZE = 800000
    series_sizes = [s.data.size for s in series]
    total_size = sum(series_sizes)
    if total_size > _MAX_SIZE:
        # we need to downscale
        base_size = _MAX_SIZE / len(series_sizes)
        baseused_size = sum([min(s, base_size) for s in series_sizes])
        leftover = _MAX_SIZE - baseused_size
        for s in series:
            # if we need to down-sample, use low-pass average filter and sampling
            if s.data.size >= base_size:
                budget = int(leftover * s.data.size/(total_size-baseused_size))
                step = int(np.ceil(s.data.size / float(budget)))
                x = s.data[:, 0][::-step][::-1]
                y = s.data[:, 1]
                y_low_pass = np.convolve(y, np.ones(shape=(step,), dtype=y.dtype)/float(step), mode='same')
                y = y_low_pass[::-step][::-1]
                s.data = np.array([x, y], dtype=s.data.dtype).T

            # decide on number of points between mean and max
            s_max = np.max(np.abs(s.data), axis=0)
            digits = np.maximum(np.array([1, 1]), np.array([6, 6]) - np.floor(np.abs(np.log10(s_max))))
            s.data[:, 0] = np.round(s.data[:, 0] * (10 ** digits[0])) / (10 ** digits[0])
            s.data[:, 1] = np.round(s.data[:, 1] * (10 ** digits[1])) / (10 ** digits[1])

    plotly_obj["data"].extend({
        "name": s.name,
        "x": s.data[:, 0].tolist(),
        "y": s.data[:, 1].tolist(),
        "mode": mode,
        "text": s.labels,
        "type": "scatter",
    } for s in series)

    return plotly_obj


def create_2d_scatter_series(np_row_wise, title="Scatter", series_name="Series", xtitle="x", ytitle="y", mode="lines",
                             labels=None, comment=None):
    """
    Create a 2D scatter Plotly graph from a 2 column numpy array
    :param np_row_wise: 2 column numpy data array [(x0,y0), (x1,y1) ...]
    :param title: Chart title
    :param series_name: Series name
    :param xtitle: X-axis title
    :param ytitle: Y-axis title
    :param mode: scatter type mode ('lines' / 'markers' / 'lines+markers')
    :param labels: label (text) per point on the scatter graph
    :param comment: comment underneath the title
    :return: Plotly chart dict
    :return:
    """
    plotly_obj = _plotly_scatter_layout_dict(title=title, xaxis_title=xtitle, yaxis_title=ytitle, comment=comment)
    assert np_row_wise.ndim == 2, "Expected a 2D numpy array"
    assert np_row_wise.shape[1] == 2, "Expected two columns X/Y e.g. [(x0,y0), (x1,y1) ...]"

    this_scatter_data = {
        "name": series_name,
        "x": np_row_wise[:, 0].tolist(),
        "y": np_row_wise[:, 1].tolist(),
        "mode": mode,
        "text": labels,
        "type": "scatter"
    }
    plotly_obj["data"].append(this_scatter_data)
    return plotly_obj


def create_3d_scatter_series(np_row_wise, title="Scatter", series_name="Series", xtitle="x", ytitle="y", ztitle="z",
                             mode="lines", color=((217, 217, 217, 0.14),), marker_size=5, line_width=0.8,
                             labels=None, fill_axis=-1, plotly_obj=None):
    """
    Create a 3D scatter Plotly graph from a 3 column numpy array
    :param np_row_wise: 3 column numpy data array [(x0,y0,z0), (x1,y1,z1) ...]
    :param title: Chart title
    :param series_name: Series name
    :param xtitle: X-axis title
    :param ytitle: Y-axis title
    :param ztitle: Z-axis title
    :param labels: label (text) per point on the scatter graph
    :param fill_axis: fill area under the curve
    :return: Plotly chart dict
    :return:
    """
    if not plotly_obj:
        plotly_obj = plotly_scatter3d_layout_dict(title=title, xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle)
    assert np_row_wise.ndim == 2, "Expected a 2D numpy array"
    assert np_row_wise.shape[1] == 3, "Expected three columns X/Y/Z e.g. [(x0,y0,z0), (x1,y1,z1) ...]"

    c = color[0]
    c = (int(c[0]), int(c[1]), int(c[2]), float(c[3]))
    this_scatter_data = {
        "name": series_name,
        "x": np_row_wise[:, 0].tolist(),
        "y": np_row_wise[:, 1].tolist(),
        "z": np_row_wise[:, 2].tolist(),
        "text": labels,
        "type": "scatter3d",
        "mode": mode,
        'marker': {
            'size': marker_size,
            'line': {
                'color': 'rgba(%d, %d, %d, %f.2)' % (c[0], c[1], c[2], c[3]),
                'width': line_width
            },
            'opacity': 0.8
        },
    }
    plotly_obj["data"].append(this_scatter_data)
    return plotly_obj


def create_value_matrix(np_value_matrix, title="Heatmap Matrix", xlabels=None, ylabels=None, xtitle="X", ytitle="Y",
                        custom_colors=True, series=None, comment=None):
    conf_matrix_plot = {
        "data": [
            {
                "x": xlabels,
                "y": ylabels,
                "z": np_value_matrix.tolist(),
                "type": "heatmap"
            }
        ],
        "layout": {
            "showlegend": True,
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),

            "xaxis": {
                "title": xtitle,
            },
            "yaxis": {
                "title": ytitle
            },
            "name": series,
        }
    }

    if custom_colors:
        scale, bar = _get_z_colorbar_data()
        conf_matrix_plot["data"][0].update({"colorscale": scale})
        conf_matrix_plot["data"][0].update({"colorbar": bar})

    return conf_matrix_plot


def create_3d_surface(np_value_matrix, title="3D Surface", xlabels=None, ylabels=None, xtitle="X", ytitle="Y",
                      ztitle="Z", custom_colors=True, series=None, camera=None, comment=None):
    conf_matrix_plot = {
        "data": [
            {
                "z": np_value_matrix.tolist(),
                "type": "surface",
                "contours": {
                    "y": {
                        "show": False,
                        "highlightcolor": "#fff4ff",
                        "project": {"y": True}
                    }
                },
                "showscale": False,
            }
        ],
        "layout": {
            "scene": {
                "xaxis": {
                    "title": xtitle,
                    "showgrid": False,
                    "nticks": 10,
                    "ticktext": xlabels,
                    "tickvals": list(range(len(xlabels))) if xlabels else None,
                },
                "yaxis": {
                    "title": ytitle,
                    "showgrid": False,
                    "nticks":  10,
                    "ticktext": ylabels,
                    "tickvals": list(range(len(ylabels))) if ylabels else ylabels,
                },
                "zaxis": {
                    "title": ztitle,
                    "nticks": 5,
                },
            },
            "showlegend": False,
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),
            "name": series,
        }
    }
    if camera:
        conf_matrix_plot['layout']['scene']['camera'] = {"eye": {"x": camera[0], "y": camera[1], "z": camera[2]}}

    if custom_colors:
        scale, bar = _get_z_colorbar_data()
        conf_matrix_plot["data"][0].update({"colorscale": scale})
        conf_matrix_plot["data"][0].update({"colorbar": bar})

    return conf_matrix_plot


def create_image_plot(image_src, title, width=640, height=480, series=None, comment=None):
    image_plot = {
        "data": [],
        "layout": {
            "xaxis": {"visible": False, "range": [0, width]},
            "yaxis": {"visible": False, "range": [0, height]},
            # "width": width,
            # "height": height,
            "margin": {'l': 0, 'r': 0, 't': 0, 'b': 0},
            "images": [{
                "sizex": width,
                "sizey": height,
                "xref": "x",
                "yref": "y",
                "opacity": 1.0,
                "x": 0,
                "y": int(height / 2),
                "yanchor": "middle",
                "sizing": "contain",
                "layer": "below",
                "source": image_src
            }],
            "showlegend": False,
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),
            "name": series,
        }
    }
    return image_plot


def _get_z_colorbar_data(z_data=None, values=None, colors=None):
    if values is None:
        values = [0, 1. / 10, 2. / 10, 6. / 10, 9. / 10]
    if colors is None:
        colors = [(71, 17, 100), (53, 92, 140), (37, 139, 141), (66, 189, 112), (141, 314, 68), (221, 226, 24)]
    if z_data is not None:
        data = np.array(z_data)
        max_z = data.max()
        scaler = max_z
        values = [float(v * scaler) for v in values[0:5]]
    values.append(1.0)  # poltly quirk?
    # we do not want to show the first and last value
    tickvalues = [" %.3f " % v for v in values[1:]]
    tickvalues = [float(v) for v in tickvalues]
    # tickvalues.pop()
    colorscale = [[v, 'rgb' + str(color)] for v, color in zip(values, colors)]
    colorbar = {"tick0": 0, "tickmode": "array", "tickvals": tickvalues}

    return colorscale, colorbar


def _plotly_hist_dict(title, xtitle, ytitle, data=None, comment=None):
    """
    Create a basic Plotly chart dictionary
    :param title: Chart title
    :param xtitle: X-Series title
    :param ytitle: Y-Series title
    :param data: Data items
    :type data: list
    :return: Plotly chart dict
    """
    return {
        "data": data or [],
        "layout": {
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),
            "xaxis": {
                "title": xtitle
            },
            "yaxis": {
                "title": ytitle
            },
            "barmode": "stack",
            "bargap": 0.08,
            "bargroupgap": 0
        }
    }


def _np_row_to_plotly_data_item(np_row, label, xlabels=None):
    """
    Convert a numpy data row into a Plotly chart data item
    :param np_row: numpy 1D data row
    :param label: Item label
    :return: Plotly data item dict
    """
    bins = list(range(np_row.shape[0])) if xlabels is None else list(xlabels)
    # mylabels = ['"' + label + '"'] * len(bins)
    this_trace_data = {
        "name": label,
        "y": np_row.tolist(),
        "x": bins,
        # "text": mylabels,
        "type": "bar"
    }
    return this_trace_data


def _plotly_scatter_layout_dict(title="Scatter", xaxis_title="X", yaxis_title="Y", series=None, comment=None):
    return {
        "data": [],
        "layout": {
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),
            "xaxis": {
                "title": xaxis_title,
                "showspikes": True,
                "spikethickness": 1,
                "spikesnap": "cursor",
                "spikemode": "toaxis+across",
            },
            "yaxis": {
                "title": yaxis_title,
                "showspikes": True,
                "spikethickness": 1,
                "spikesnap": "cursor",
                "spikemode": "toaxis+across",
            },
            "name": series,
        }
    }


def plotly_scatter3d_layout_dict(title="Scatter", xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                                 series=None, show_legend=True, comment=None):
    return {
        "data": [],
        "layout": {
            "showlegend": show_legend,
            "title": title if not comment else (title + '<br><sup>' + comment + '</sup>'),
            "scene": {
                'xaxis': {'title': xaxis_title},
                'yaxis': {'title': yaxis_title},
                'zaxis': {'title': zaxis_title},
            },
            "name": series,
        }
    }
