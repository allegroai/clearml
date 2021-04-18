<div align="center">

<a href="https://app.community.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/clearml-logo.svg?raw=true" width="250px"></a>


**ClearML - Auto-Magical Suite of tools to streamline your ML workflow 
</br>Experiment Manager, ML-Ops and Data-Management**

[![GitHub license](https://img.shields.io/github/license/allegroai/clearml.svg)](https://img.shields.io/github/license/allegroai/clearml.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/clearml.svg)](https://img.shields.io/pypi/pyversions/clearml.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/clearml.svg)](https://img.shields.io/pypi/v/clearml.svg)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![Slack Channel](https://img.shields.io/badge/slack-%23clearml--community-blueviolet?logo=slack)](https://join.slack.com/t/clearml/shared_invite/zt-c0t13pty-aVUZZW1TSSSg2vyIGVPBhg)

</div>

---
### ClearML
#### *Formerly known as Allegro Trains*
ClearML is a ML/DL development and production suite, it contains three main modules:

- [Experiment Manager](#clearml-experiment-manager) - Automagical experiment tracking, environments and results
- [ML-Ops](https://github.com/allegroai/clearml-agent) - Automation, Pipelines & Orchestration solution for ML/DL jobs (K8s / Cloud / bare-metal)  
- [Data-Management](https://github.com/allegroai/clearml/blob/master/docs/datasets.md) - Fully differentiable data management & version control solution on top of object-storage 
  (S3/GS/Azure/NAS)  
  

Instrumenting these components is the **ClearML-server**, see [Self-Hosting](https://allegro.ai/clearml/docs/rst/deploying_clearml/index.html) & [Free tier Hosting](https://app.community.clear.ml)  


---
<div align="center">

**[Sign up](https://app.community.clear.ml)  &  [Start using](https://allegro.ai/clearml/docs/rst/getting_started/index.html) in under 2 minutes**  

</div>

---
<a href="https://app.community.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/webapp_screenshots.gif?raw=true" width="100%"></a>

## ClearML Experiment Manager

**Adding only 2 lines to your code gets you the following**

* Complete experiment setup log
    * Full source control info including non-committed local changes
    * Execution environment (including specific packages & versions)
    * Hyper-parameters
        * ArgParser for command line parameters with currently used values
        * Explicit parameters dictionary
        * Tensorflow Defines (absl-py)
        * Hydra configuration and overrides
    * Initial model weights file
* Full experiment output automatic capture
    * stdout and stderr
    * Resource Monitoring (CPU/GPU utilization, temperature, IO, network, etc.)
    * Model snapshots (With optional automatic upload to central storage: Shared folder, S3, GS, Azure, Http)
    * Artifacts log & store (Shared folder, S3, GS, Azure, Http)
    * Tensorboard/TensorboardX scalars, metrics, histograms, **images, audio and video samples**
    * [Matplotlib & Seaborn](https://github.com/allegroai/clearml/tree/master/examples/frameworks/matplotlib)
    * [ClearML Explicit Logging](https://allegro.ai/clearml/docs/docs/tutorials/tutorial_explicit_reporting.html#step-2-logger-class-reporting-methods) interface for complete flexibility.
* Extensive platform support and integrations
    * Supported ML/DL frameworks: [PyTorch](https://github.com/allegroai/clearml/tree/master/examples/frameworks/pytorch)(incl' ignite/lightning), [Tensorflow](https://github.com/allegroai/clearml/tree/master/examples/frameworks/tensorflow), [Keras](https://github.com/allegroai/clearml/tree/master/examples/frameworks/keras), [AutoKeras](https://github.com/allegroai/clearml/tree/master/examples/frameworks/autokeras), [XGBoost](https://github.com/allegroai/clearml/tree/master/examples/frameworks/xgboost) and [Scikit-Learn](https://github.com/allegroai/clearml/tree/master/examples/frameworks/scikit-learn)
    * Seamless integration (including version control) with **Jupyter Notebook**
    and [*PyCharm* remote debugging](https://github.com/allegroai/trains-pycharm-plugin)
      
#### [Start using ClearML](https://allegro.ai/clearml/docs/rst/getting_started/index.html) 

```bash
pip install clearml
```

Add two lines to your code:
```python
from clearml import Task
task = Task.init(project_name='examples', task_name='hello world')
```

You are done, everything your process outputs is now automagically logged into ClearML.
<br>Next step automation! **Learn more on ClearML two clicks automation [here](https://allegro.ai/clearml/docs/rst/clearml_agent/index.html)** 

## ClearML Architecture

The ClearML run-time components:

* The ClearML Python Package for integrating ClearML into your existing scripts by adding just two lines of code, and optionally extending your experiments and other workflows with ClearML powerful and versatile set of classes and methods.
* The ClearML Server storing experiment, model, and workflow data, and supporting the Web UI experiment manager, and ML-Ops automation for reproducibility and tuning. It is available as a hosted service and open source for you to deploy your own ClearML Server.
* The ClearML Agent for ML-Ops orchestration, experiment and workflow reproducibility, and scalability.

<img src="https://allegro.ai/clearml/docs/_images/ClearML_Architecture.png" width="100%" alt="clearml-architecture">

## Additional Modules 

- [clearml-session](https://github.com/allegroai/clearml-session) - **Launch remote JupyterLab / VSCode-server inside any docker, on Cloud/On-Prem machines**
- [clearml-task](https://github.com/allegroai/clearml/blob/master/docs/clearml-task.md) - Run any codebase on remote machines with full remote logging of Tensorboard, Matplotlib & Console outputs 
- [clearml-data](https://github.com/allegroai/clearml/blob/master/docs/datasets.md) - **CLI for managing and versioning your datasets, including creating / uploading / downloading of data from S3/GS/Azure/NAS** 
- [AWS Auto-Scaler](https://allegro.ai/clearml/docs/docs/examples/services/aws_autoscaler/aws_autoscaler.html) - Automatically spin EC2 instances based on your workloads with preconfigured budget! No need for K8s!
- [Hyper-Parameter Optimization](https://allegro.ai/clearml/docs/docs/examples/frameworks/pytorch/notebooks/image/hyperparameter_search.html) - Optimize any code with black-box approach and state of the art Bayesian optimization algorithms 
- [Automation Pipeline](https://allegro.ai/clearml/docs/docs/examples/frameworks/pytorch/notebooks/table/tabular_training_pipeline.html) - Build pipelines based on existing experiments / jobs, supports building pipelines of pipelines!  
- [Slack Integration](https://allegro.ai/clearml/docs/docs/examples/services/monitoring/slack_alerts.html) - Report experiments progress / failure directly to Slack (fully customizable!)  

## Why ClearML?

ClearML is our solution to a problem we share with countless other researchers and developers in the machine
learning/deep learning universe: Training production-grade deep learning models is a glorious but messy process.
ClearML tracks and controls the process by associating code version control, research projects,
performance metrics, and model provenance.

We designed ClearML specifically to require effortless integration so that teams can preserve their existing methods
and practices. 

  - Use it on a daily basis to boost collaboration and visibility in your team 
  - Create a remote job from any experiment with a click of a button
  - Automate processes and create pipelines to collect your experimentation logs, outputs, and data
  - Store all you data on any object-storage solution, with the simplest interface possible
  - Make you data transparent by cataloging it all on the ClearML platform    

We believe ClearML is ground-breaking. We wish to establish new standards of true seamless integration between
experiment management,ML-Ops and data management. 

## Who We Are

ClearML is supported by the team behind *allegro.ai*,
where we build deep learning pipelines and infrastructure for enterprise companies.

We built ClearML to track and control the glorious but messy process of training production-grade deep learning models.
We are committed to vigorously supporting and expanding the capabilities of ClearML.

We promise to always be backwardly compatible, making sure all your logs, data and pipelines 
will always upgrade with you.

## License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)

## Documentation, Community & Support

More information in the [official documentation](https://allegro.ai/clearml/docs) and [on YouTube](https://www.youtube.com/c/ClearML).

For examples and use cases, check the [examples folder](https://github.com/allegroai/clearml/tree/master/examples) and [corresponding documentation](https://allegro.ai/clearml/docs/rst/examples/index.html).

If you have any questions: post on our [Slack Channel](https://join.slack.com/t/clearml/shared_invite/zt-c0t13pty-aVUZZW1TSSSg2vyIGVPBhg), or tag your questions on [stackoverflow](https://stackoverflow.com/questions/tagged/clearml) with '**[clearml](https://stackoverflow.com/questions/tagged/clearml)**' tag (*previously [trains](https://stackoverflow.com/questions/tagged/trains) tag*).

For feature requests or bug reports, please use [GitHub issues](https://github.com/allegroai/clearml/issues).

Additionally, you can always find us at *clearml@allegro.ai*

## Contributing

**PRs are always welcomed** :heart: See more details in the ClearML [Guidelines for Contributing](https://github.com/allegroai/clearml/blob/master/docs/contributing.md).


_May the force (and the goddess of learning rates) be with you!_
