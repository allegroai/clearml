# ClearML - Example of manual graphs and  statistics reporting
#
import math

import numpy as np
from bokeh.models import ColumnDataSource, GraphRenderer, Oval, StaticLayoutProvider
from bokeh.palettes import Spectral5, Spectral8
from bokeh.plotting import figure, output_file, save
from bokeh.sampledata.autompg import autompg_clean as bokeh_df
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

from clearml import Task, Logger


def report_html_url(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting html from url to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    logger.report_media("html", "url_html", iteration=iteration, url="https://allegro.ai/docs/index.html")


def report_html_periodic_table(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting interactive (html) of periodic table to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    output_file("periodic.html")
    periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
    groups = [str(x) for x in range(1, 19)]
    autompg_clean = elements.copy()
    autompg_clean["atomic mass"] = autompg_clean["atomic mass"].astype(str)
    autompg_clean["group"] = autompg_clean["group"].astype(str)
    autompg_clean["period"] = [periods[x - 1] for x in autompg_clean.period]
    autompg_clean = autompg_clean[autompg_clean.group != "-"]
    autompg_clean = autompg_clean[autompg_clean.symbol != "Lr"]
    autompg_clean = autompg_clean[autompg_clean.symbol != "Lu"]
    cmap = {
        "alkali metal": "#a6cee3",
        "alkaline earth metal": "#1f78b4",
        "metal": "#d93b43",
        "halogen": "#999d9a",
        "metalloid": "#e08d49",
        "noble gas": "#eaeaea",
        "nonmetal": "#f1d4Af",
        "transition metal": "#599d7A",
    }
    source = ColumnDataSource(autompg_clean)
    p = figure(
        plot_width=900,
        plot_height=500,
        title="Periodic Table (omitting LA and AC Series)",
        x_range=groups,
        y_range=list(reversed(periods)),
        toolbar_location=None,
        tools="hover",
    )
    p.rect(
        "group",
        "period",
        0.95,
        0.95,
        source=source,
        fill_alpha=0.6,
        legend_label="metal",
        color=factor_cmap(
            "metal", palette=list(cmap.values()), factors=list(cmap.keys())
        ),
    )
    text_props = {"source": source, "text_align": "left", "text_baseline": "middle"}
    x = dodge("group", -0.4, range=p.x_range)
    r = p.text(x=x, y="period", text="symbol", **text_props)
    r.glyph.text_font_style = "bold"
    r = p.text(
        x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number", **text_props
    )
    r.glyph.text_font_size = "8pt"
    r = p.text(
        x=x, y=dodge("period", -0.35, range=p.y_range), text="name", **text_props
    )
    r.glyph.text_font_size = "5pt"
    r = p.text(
        x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass", **text_props
    )
    r.glyph.text_font_size = "5pt"
    p.text(
        x=["3", "3"],
        y=["VI", "VII"],
        text=["LA", "AC"],
        text_align="center",
        text_baseline="middle",
    )
    p.hover.tooltips = [
        ("Name", "@name"),
        ("Atomic number", "@{atomic number}"),
        ("Atomic mass", "@{atomic mass}"),
        ("Type", "@metal"),
        ("CPK color", "$color[hex, swatch]:CPK"),
        ("Electronic configuration", "@{electronic configuration}"),
    ]
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    save(p)
    logger.report_media("html", "periodic_html", iteration=iteration, local_path="periodic.html")


def report_html_groupby(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting bokeh groupby (html) to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    output_file("bar_pandas_groupby_nested.html")
    bokeh_df.cyl = bokeh_df.cyl.astype(str)
    bokeh_df.yr = bokeh_df.yr.astype(str)
    group = bokeh_df.groupby(by=["cyl", "mfr"])
    index_cmap = factor_cmap(
        "cyl_mfr", palette=Spectral5, factors=sorted(bokeh_df.cyl.unique()), end=1
    )
    p = figure(
        plot_width=800,
        plot_height=300,
        title="Mean MPG by # Cylinders and Manufacturer",
        x_range=group,
        toolbar_location=None,
        tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")],
    )
    p.vbar(
        x="cyl_mfr",
        top="mpg_mean",
        width=1,
        source=group,
        line_color="white",
        fill_color=index_cmap,
    )
    p.y_range.start = 0
    p.x_range.range_padding = 0.05
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
    p.xaxis.major_label_orientation = 1.2
    p.outline_line_color = None
    save(p)
    logger.report_media(
        "html",
        "pandas_groupby_nested_html",
        iteration=iteration,
        local_path="bar_pandas_groupby_nested.html",
    )


def report_html_graph(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting bokeh graph (html) to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    nodes = 8
    node_indices = list(range(nodes))
    plot = figure(
        title="Graph Layout Demonstration",
        x_range=(-1.1, 1.1),
        y_range=(-1.1, 1.1),
        tools="",
        toolbar_location=None,
    )
    graph = GraphRenderer()
    graph.node_renderer.data_source.add(node_indices, "index")
    graph.node_renderer.data_source.add(Spectral8, "color")
    graph.node_renderer.glyph = Oval(height=0.1, width=0.2, fill_color="color")
    graph.edge_renderer.data_source.data = dict(start=[0] * nodes, end=node_indices)
    # start of layout code
    circ = [i * 2 * math.pi / 8 for i in node_indices]
    x = [math.cos(i) for i in circ]
    y = [math.sin(i) for i in circ]
    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    plot.renderers.append(graph)
    output_file("graph.html")
    save(plot)
    logger.report_media("html", "Graph_html", iteration=iteration, local_path="graph.html")


def report_html_image(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting bokeh image (html) to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """

    # First html
    samples = 500
    x = np.linspace(0, 10, samples)
    y = np.linspace(0, 10, samples)
    xx, yy = np.meshgrid(x, y)
    d = np.sin(xx) * np.cos(yy)
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    p.x_range.range_padding = p.y_range.range_padding = 0
    # must give a vector of image data for image parameter
    p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
    p.grid.grid_line_width = 0.5
    output_file("image.html", title="image.py example")
    save(p)
    logger.report_media("html", "Spectral_html", iteration=iteration, local_path="image.html")


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="html samples reporting")

    print('reporting html files into debug samples section')

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report html as debug samples
    report_html_image(logger)
    report_html_graph(logger)
    report_html_groupby(logger)
    report_html_periodic_table(logger)
    report_html_url(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
