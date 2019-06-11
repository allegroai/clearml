# TRAINS
## Magic Version Control & Experiment Manager for AI

<p style="font-size:1.2rem; font-weight:700;">"Because it’s a jungle out there"</p>

[![GitHub license](https://img.shields.io/github/license/allegroai/trains.svg)](https://img.shields.io/github/license/allegroai/trains.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/trains.svg)](https://img.shields.io/pypi/pyversions/trains.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/trains.svg)](https://img.shields.io/pypi/v/trains.svg)
[![PyPI status](https://img.shields.io/pypi/status/trains.svg)](https://pypi.python.org/pypi/trains/)

Behind every great scientist are great repeatable methods.  Sadly, this is easier said than done.

When talented scientists, engineers, or developers work on their own, a mess may be unavoidable. Yet, it may still be
manageable. However, with time and more people joining your project,
managing the clutter takes its toll on productivity.
As your project moves toward production,
visibility and provenance for scaling your deep-learning efforts are a must, but both
suffer as your team grows.

For teams or entire companies, TRAINS logs everything in one central server and takes on the responsibilities for visibility and provenance
so productivity does not suffer.
TRAINS records and manages various deep learning research workloads and does so with unbelievably small integration costs.

TRAINS is an auto-magical experiment manager that you can use productively with minimal integration and while
preserving your existing methods and practices. Use it on a daily basis to boost collaboration and visibility,
or use it to automatically collect your experimentation logs, outputs, and data to one centralized server for provenance.

(See TRAINS live at [https://demoapp.trainsai.io](https://demoapp.trainsai.io))
![Alt Text](https://github.com/allegroai/trains/blob/master/docs/webapp_screenshots.gif?raw=true)

## Why Should I Use TRAINS?

TRAINS is our solution to a problem we share with countless other researchers and developers in the
machine learning/deep learning universe.
Training production-grade deep learning models is a glorious but messy process.
We built TRAINS to solve that problem. TRAINS tracks and controls the process by associating code version control, research projects, performance metrics, and model provenance.
TRAINS removes the mess but leaves the glory.


Choose TRAINS because...

* Sharing experiments with the team is difficult and gets even more difficult further up the chain.
* Like all of us, you lost a model and are left with no repeatable process.
* You setup up a central location for TensorBoard and it exploded with a gazillion experiments.
* You accidentally threw away important results while trying to manually clean up the clutter.
* You do not associate the train code commit with the model or TensorBoard logs.
* You are storing model parameters in the checkpoint filename.
* You cannot find any other tool for comparing results, hyper-parameters and code commits.
* TRAINS requires **only two-lines of code** for full integration.
* TRAINS is **free**.

## Main Features

* Seamless integration with leading frameworks, including: PyTorch, TensorFlow, Keras, and others coming soon!
* Track everything with two lines of code.
* Model logging that automatically associates models with code and the parameters used to train them, including initial weights logging.
* Multi-user process tracking and collaboration.
* **Experiment comparison** including code commits, initial weights, hyper-parameters and metric results.
* Management capabilities including project management, filter-by-metric.
* Centralized server for aggregating logs, records, and general bookkeeping.
* Automatically create a copy of models on centralized storage (TRAINS supports shared folders, S3, GS, and Azure is coming soon!).
* Support for Jupyter notebook (see the [trains-jupyter-plugin](https://github.com/allegroai/trains-jupyter-plugin)) and PyCharm remote debugging (see the [trains-pycharm-plugin](https://github.com/allegroai/trains-pycharm-plugin)).
* A field-tested, feature-rich SDK for your on-the-fly customization needs.


## TRAINS Magically Logs

TRAINS magically logs the following:

* Git repository, branch and commit id
* Hyper-parameters, including:
    * ArgParser for command line parameters with currently used values
    * Tensorflow Defines (absl-py)
    * Manually passed parameter dictionary
* Initial model weights file
* Model snapshots
* stdout and stderr
* TensorBoard scalars, metrics, histograms, images, and audio coming soon (also tensorboardX)
* Matplotlib

## See for Yourself

We have a demo server up and running [https://demoapp.trainsai.io](https://demoapp.trainsai.io) (it resets every 24 hours and all of the data is deleted).

You can test your code with it:

1. Install TRAINS

        pip install trains

1. Add the following to your code:

        from trains import Task
        Task = Task.init(project_name=”my_projcet”, task_name=”my_task”)

1. Run your code. When TRAINS connects to the server, a link prints. For example:

        TRAINS Metrics page:
        https://demoapp.trainsai.io/projects/76e5e2d45e914f52880621fe64601e85/experiments/241f06ae0f5c4b27b8ce8b64890ce152/output/log

1. Open your link and view the experiment parameters, model and tensorboard metrics.


## How TRAINS Works

TRAINS is composed of the following:

* [TRAINS-server](https://github.com/allegroai/trains-server)
* [Web-App](https://github.com/allegroai/trains-web) (web user interface)
* Python SDK (auto-magically connects your code, see [Using TRAINS](#using-trains-example))

The following diagram illustrates the interaction of the TRAINS-server and a GPU machine:

<pre>
    TRAINS-server
    
    +--------------------------------------------------------------------+
    |                                                                    |
    |   Server Docker                   Elastic Docker     Mongo Docker  |
    |  +-------------------------+     +---------------+  +------------+ |
    |  |     Pythonic Server     |     |               |  |            | |
    |  |   +-----------------+   |     | ElasticSearch |  |  MongoDB   | |
    |  |   |   WEB server    |   |     |               |  |            | |
    |  |   |   Port 8080     |   |     |               |  |            | |
    |  |   +--------+--------+   |     |               |  |            | |
    |  |            |            |     |               |  |            | |
    |  |   +--------+--------+   |     |               |  |            | |
    |  |   |   API server    +----------------------------+            | |
    |  |   |   Port 8008     +---------+               |  |            | |
    |  |   +-----------------+   |     +-------+-------+  +-----+------+ |
    |  |                         |             |                |        |
    |  |   +-----------------+   |         +---+----------------+------+ |
    |  |   |   File Server   +-------+     |    Host Storage           | |
    |  |   |   Port 8081     |   |   +-----+                           | |
    |  |   +-----------------+   |         +---------------------------+ |
    |  +------------+------------+                                       |
    +---------------|----------------------------------------------------+
                    |HTTP
                    +--------+
    GPU Machine              |
    +------------------------|-------------------------------------------+
    |     +------------------|--------------+                            |
    |     |  Training        |              |    +---------------------+ |
    |     |  Code        +---+------------+ |    | TRAINS configuration| |
    |     |              | TRAINS         | |    | ~/trains.conf       | |
    |     |              |                +------+                     | |
    |     |              +----------------+ |    +---------------------+ |
    |     +---------------------------------+                            |
    +--------------------------------------------------------------------+
</pre>

## Installing and Configuring TRAINS

1. Install the trains-server docker (see [Installing the TRAINS Server](https://github.com/allegroai/trains-server)).

1. Install the TRAINS package:

    	pip install trains

1. Run the initial configuration wizard to setup the trains-server (ip:port and user credentials):

	    trains-init

After installing and configuring, your configuration is `~/trains.conf`. View a sample configuration file [here](https://github.com/allegroai/trains/blob/master/docs/trains.conf).

## Using TRAINS

Add these two lines of to your code:

    from trains import Task
    task = Task.init(project_name, task_name)

* If no project name is provided, then the repository name is used.
* If no task (experiment) name is provided, then the main filename is used as experiment name

Executing your script prints a direct link to the currently running experiment page, for example:

```bash
TRAINS Metrics page:

https://demoapp.trainsai.io/projects/76e5e2d45e914f52880621fe64601e85/experiments/241f06ae0f5c4b27b8ce8b64890ce152/output/log
```

![Alt Text](https://github.com/allegroai/trains/blob/master/docs/results_screenshots.gif?raw=true)

For more examples and use cases, see [examples](https://github.com/allegroai/trains/tree/master/examples).

## Who Supports TRAINS?

The people behind *allegro.ai*.
We build deep learning pipelines and infrastructure for enterprise companies.
We built TRAINS to track and control the glorious
but messy process of training production-grade deep learning models.
We are committed to vigorously supporting and expanding the capabilities of TRAINS,
because it is not only our beloved creation, we also use it daily.

## Why Are We Releasing TRAINS?

We believe TRAINS is ground-breaking. We wish to establish new standards of experiment management in
machine- and deep-learning.
Only the greater community can help us do that.

We promise to always be backwardly compatible. If you start working with TRAINS today, even though this code is still in the beta stage, your logs and data will always upgrade with you.

## License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)

## Guidelines for Contributing

See the TRAINS [Guidelines for Contributing](https://github.com/allegroai/trains/blob/master/docs/contributing.md).

## FAQ

See the TRAINS [FAQ](https://github.com/allegroai/trains/blob/master/docs/faq.md).

<p style="font-size:0.9rem; font-weight:700; font-style:italic">May the force (and the goddess of learning rates) be with you!</p>

