# TRAINS
## Auto-Magical Experiment Manager & Version Control for AI

<p style="font-size:1.2rem; font-weight:700;">"Because it’s a jungle out there"</p>

[![GitHub license](https://img.shields.io/github/license/allegroai/trains.svg)](https://img.shields.io/github/license/allegroai/trains.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/trains.svg)](https://img.shields.io/pypi/pyversions/trains.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/trains.svg)](https://img.shields.io/pypi/v/trains.svg)
[![PyPI status](https://img.shields.io/pypi/status/trains.svg)](https://pypi.python.org/pypi/trains/)

Behind every great scientist are great repeatable methods. Sadly, this is easier said than done.

When talented scientists, engineers, or developers work on their own, a mess may be unavoidable.
Yet, it may still be manageable. However, with time and more people joining your project, managing the clutter takes
its toll on productivity. As your project moves toward production, visibility and provenance for scaling your
deep-learning efforts are a must.

For teams or entire companies, TRAINS logs everything in one central server and takes on the responsibilities for
visibility and provenance so productivity does not suffer. TRAINS records and manages various deep learning
research workloads and does so with practically zero integration costs.

We designed TRAINS specifically to require effortless integration so that teams can preserve their existing methods
and practices. Use it on a daily basis to boost collaboration and visibility, or use it to automatically collect
your experimentation logs, outputs, and data to one centralized server.

(See TRAINS live at [https://demoapp.trainsai.io](https://demoapp.trainsai.io))
![Alt Text](https://github.com/allegroai/trains/blob/master/docs/webapp_screenshots.gif?raw=true)


## Main Features

TRAINS is our solution to a problem we shared with countless other researchers and developers in the machine
learning/deep learning universe: Training production-grade deep learning models is a glorious but messy process.
TRAINS tracks and controls the process by associating code version control, research projects,
performance metrics, and model provenance.

* Start today!
    * TRAINS is free and open-source
    * TRAINS requires only two lines of code for full integration
* Use it with your favorite tools
    * Seamless integration with leading frameworks, including: *PyTorch*, *TensorFlow*, *Keras*, and others coming soon
    * Support for *Jupyter Notebook* (see [trains-jupyter-plugin](https://github.com/allegroai/trains-jupyter-plugin))
    and *PyCharm* remote debugging (see [trains-pycharm-plugin](https://github.com/allegroai/trains-pycharm-plugin))
* Log everything. Experiments become truly repeatable
    * Model logging with **automatic association** of **model + code + parameters + initial weights**
    * Automatically create a copy of models on centralized storage 
    ([supports shared folders, S3, GS,](https://github.com/allegroai/trains/blob/master/docs/faq.md#i-read-there-is-a-feature-for-centralized-model-storage-how-do-i-use-it-) and Azure is coming soon!)
* Share and collaborate
    * Multi-user process tracking and collaboration
    * Centralized server for aggregating logs, records, and general bookkeeping
* Increase productivity
    * Comprehensive **experiment comparison**: code commits, initial weights, hyper-parameters and metric results
* Order & Organization
    * Manage and organize your experiments in projects
    * Query capabilities; sort and filter experiments by results metrics
* And more
    * Stop an experiment on a remote machine using the web-app
    * A field-tested, feature-rich SDK for your on-the-fly customization needs


## TRAINS Automatically Logs

* Git repository, branch, commit id and entry point (git diff coming soon)
    * Hyper-parameters, including
    * ArgParser for command line parameters with currently used values
    * Tensorflow Defines (absl-py)
* Explicit parameters dictionary
* Initial model weights file
* Model snapshots
* stdout and stderr
* Tensorboard/TensorboardX scalars, metrics, histograms, images (with audio coming soon)
* Matplotlib


## See for Yourself

We have a demo server up and running at https://demoapp.trainsai.io. You can try out TRAINS and test your code with it.
Note that it resets every 24 hours and all of the data is deleted.

Connect your code with TRAINS:

1. Install TRAINS

        pip install trains

1. Add the following lines to your code

        from trains import Task
        task = Task.init(project_name="my project", task_name="my task")

1. Run your code. When TRAINS connects to the server, a link is printed. For example

        TRAINS Results page:
        https://demoapp.trainsai.io/projects/76e5e2d45e914f52880621fe64601e85/experiments/241f06ae0f5c4b27b8ce8b64890ce152/output/log

1. Open the link and view your experiment parameters, model and tensorboard metrics


## How TRAINS Works

TRAINS is a two part solution:

1. TRAINS [python package](https://pypi.org/project/trains/) (auto-magically connects your code, see [Using TRAINS](#using-trains))
2. [TRAINS-server](https://github.com/allegroai/trains-server) for logging, querying, control and UI ([Web-App](https://github.com/allegroai/trains-web))

The following diagram illustrates the interaction of the [TRAINS-server](https://github.com/allegroai/trains-server)
and a GPU training machine using the TRAINS python package

<!---
![Alt Text](https://github.com/allegroai/trains/blob/master/docs/system_diagram.png?raw=true)
-->
<img src="https://github.com/allegroai/trains/blob/master/docs/system_diagram.png?raw=true" width="50%">


## Installing and Configuring TRAINS

1. Install and run trains-server (see [Installing the TRAINS Server](https://github.com/allegroai/trains-server))

2. Install TRAINS package

    	pip install trains

3. Run the initial configuration wizard and follow the instructions to setup TRAINS package
(http://**_trains-server ip_**:__port__ and user credentials)

	    trains-init

After installing and configuring, you can access your configuration file at `~/trains.conf`

Sample configuration file available [here](https://github.com/allegroai/trains/blob/master/docs/trains.conf).

## Using TRAINS

Add the following two lines to the beginning of your code

    from trains import Task
    task = Task.init(project_name, task_name)

* If project_name is not provided, the repository name will be used instead
* If task_name (experiment) is not provided, the current filename will be used instead

Executing your script prints a direct link to the experiment results page, for example:

```bash
TRAINS Results page:

https://demoapp.trainsai.io/projects/76e5e2d45e914f52880621fe64601e85/experiments/241f06ae0f5c4b27b8ce8b64890ce152/output/log
```

*For more examples and use cases*, see [examples](https://github.com/allegroai/trains/blob/master/docs/trains_examples.md).

![Alt Text](https://github.com/allegroai/trains/blob/master/docs/results_screenshots.gif?raw=true)


## Who Supports TRAINS?

TRAINS is supported by the same team behind *allegro.ai*,
where we build deep learning pipelines and infrastructure for enterprise companies.

We built TRAINS to track and control the glorious but messy process of training production-grade deep learning models.
We are committed to vigorously supporting and expanding the capabilities of TRAINS.

## Why Are We Releasing TRAINS?

We believe TRAINS is ground-breaking. We wish to establish new standards of experiment management in
deep-learning and ML. Only the greater community can help us do that.

We promise to always be backwardly compatible. If you start working with TRAINS today,
even though this project is currently in the beta stage, your logs and data will always upgrade with you.

## License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)

## Guidelines for Contributing

See the TRAINS [Guidelines for Contributing](https://github.com/allegroai/trains/blob/master/docs/contributing.md).

## FAQ

See the TRAINS [FAQ](https://github.com/allegroai/trains/blob/master/docs/faq.md).

<p style="font-size:0.9rem; font-weight:700; font-style:italic">May the force (and the goddess of learning rates) be with you!</p>

