# TRAINS
## Auto-Magical Experiment Manager & Version Control for AI

"Because it’s a jungle out there"

[![GitHub license](https://img.shields.io/github/license/allegroai/trains.svg)](https://img.shields.io/github/license/allegroai/trains.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/trains.svg)](https://img.shields.io/pypi/pyversions/trains.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/trains.svg)](https://img.shields.io/pypi/v/trains.svg)
[![PyPI status](https://img.shields.io/pypi/status/trains.svg)](https://pypi.python.org/pypi/trains/)

TRAINS is our solution to a problem we share with countless other researchers and developers in the machine
learning/deep learning universe: Training production-grade deep learning models is a glorious but messy process.
TRAINS tracks and controls the process by associating code version control, research projects,
performance metrics, and model provenance.

We designed TRAINS specifically to require effortless integration so that teams can preserve their existing methods
and practices. Use it on a daily basis to boost collaboration and visibility, or use it to automatically collect
your experimentation logs, outputs, and data to one centralized server.

(Experience TRAINS live at [https://demoapp.trainsai.io](https://demoapp.trainsai.io))
![Alt Text](https://github.com/allegroai/trains/blob/master/docs/webapp_screenshots.gif?raw=true)

## Main Features

* Seamless integration with leading frameworks, including: *PyTorch*, *TensorFlow*, *Keras*, and others coming soon
* Support for *Jupyter Notebook* and *PyCharm* remote debugging 
* Automatic log collection.
* Query, Filter, and Compare your experiment data and results
* Share and collaborate
    
**Detailed overview of TRAINS offering and system design can be found [Here](https://github.com/allegroai/trains/blob/master/docs/brief.md).**


## Using TRAINS

We have a demo server up and running at https://demoapp.trainsai.io. You can try out TRAINS and test your code with it.
Note that it resets every 24 hours and all of the data is deleted.

When you are ready to use your own TRAINS server, go ahead and [install *TRAINS-server*](#configuring-your-own-trains).

TRAINS requires only two lines of code for full integration.

To connect your code with TRAINS:

1. Install TRAINS

        pip install trains

2. Add the following lines to your code

        from trains import Task
        task = Task.init(project_name="my project", task_name="my task")

    * If project_name is not provided, the repository name will be used instead
    * If task_name (experiment) is not provided, the current filename will be used instead

3. Run your code. When TRAINS connects to the server, a link is printed. For example

        TRAINS Results page:
        https://demoapp.trainsai.io/projects/76e5e2d45e914f52880621fe64601e85/experiments/241f06ae0f5c4b27b8ce8b64890ce152/output/log

4. Open the link and view your experiment parameters, model and tensorboard metrics

## Configuring Your Own TRAINS

1. Install and run *TRAINS-server* (see [Installing the TRAINS Server](https://github.com/allegroai/trains-server))

2. Run the initial configuration wizard for your TRAINS installation and follow the instructions to setup TRAINS package
(http://**_trains-server-ip_**:__port__ and user credentials)

	    trains-init

After installing and configuring, you can access your configuration file at `~/trains.conf`

Sample configuration file available [here](https://github.com/allegroai/trains/blob/master/docs/trains.conf).


*For more examples and use cases*, see [examples](https://github.com/allegroai/trains/blob/master/docs/trains_examples.md).

## Who We Are

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

## Community

If you have any questions, look to the TRAINS [FAQ](https://github.com/allegroai/trains/blob/master/docs/faq.md), or
tag your questions on [stackoverflow](https://stackoverflow.com/questions/tagged/trains) with '**trains**' tag.

For feature requests or bug reports, please use [GitHub issues](https://github.com/allegroai/trains/issues).

Additionally, you can always find us at *trains@allegro.ai*

## Contributing

See the TRAINS [Guidelines for Contributing](https://github.com/allegroai/trains/blob/master/docs/contributing.md).


_May the force (and the goddess of learning rates) be with you!_

