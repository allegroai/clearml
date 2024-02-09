<div align="center" style="text-align: center">

<p style="text-align: center">
  <img align="center" src="docs/clearml-logo.svg#gh-light-mode-only" alt="Clear|ML"><img align="center" src="docs/clearml-logo-dark.svg#gh-dark-mode-only" alt="Clear|ML">
</p>

**[ClearML](https://clear.ml) - Auto-Magical Suite of tools to streamline your AI workflow
</br>Experiment Manager, MLOps/LLMOps and Data-Management**

[![GitHub license](https://img.shields.io/github/license/allegroai/clearml.svg)](https://img.shields.io/github/license/allegroai/clearml.svg) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/clearml.svg)](https://img.shields.io/pypi/pyversions/clearml.svg) [![PyPI version shields.io](https://img.shields.io/pypi/v/clearml.svg)](https://pypi.org/project/clearml/) [![Conda version shields.io](https://img.shields.io/conda/v/clearml/clearml)](https://anaconda.org/clearml/clearml) [![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)<br>
[![PyPI Downloads](https://static.pepy.tech/badge/clearml/month)](https://pypi.org/project/clearml/) [![Artifact Hub](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/allegroai)](https://artifacthub.io/packages/search?repo=allegroai) [![Youtube](https://img.shields.io/badge/ClearML-DD0000?logo=youtube&logoColor=white)](https://www.youtube.com/c/clearml) [![Slack Channel](https://img.shields.io/badge/slack-%23clearml--community-blueviolet?logo=slack)](https://joinslack.clear.ml) [![Signup](https://img.shields.io/badge/Clear%7CML-Signup-brightgreen)](https://app.clear.ml)

</div>

---
### ClearML
<sup>*Formerly known as Allegro Trains*<sup>

ClearML is a ML/DL development and production suite. It contains FIVE main modules:

- [Experiment Manager](#clearml-experiment-manager) - Automagical experiment tracking, environments and results
- [MLOps / LLMOps](https://github.com/allegroai/clearml-agent) - Orchestration, Automation & Pipelines solution for ML/DL/GenAI jobs (Kubernetes / Cloud / bare-metal)  
- [Data-Management](https://github.com/allegroai/clearml/blob/master/docs/datasets.md) - Fully differentiable data management & version control solution on top of object-storage 
  (S3 / GS / Azure / NAS)  
- [Model-Serving](https://github.com/allegroai/clearml-serving) - *cloud-ready* Scalable model serving solution! 
  - **Deploy new model endpoints in under 5 minutes** 
  - Includes optimized GPU serving support backed by Nvidia-Triton 
  - **with out-of-the-box  Model Monitoring** 
- [Reports](https://clear.ml/docs/latest/docs/webapp/webapp_reports) - Create and share rich MarkDown documents supporting embeddable online content 
- **NEW** :fire: [Orchestration Dashboard](https://clear.ml/docs/latest/docs/webapp/webapp_orchestration_dash/) - Live rich dashboard for your entire compute cluster (Cloud / Kubernetes / On-Prem)
  

Instrumenting these components is the **ClearML-server**, see [Self-Hosting](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server) & [Free tier Hosting](https://app.clear.ml)  


---
<div align="center">

**[Sign up](https://app.clear.ml)  &  [Start using](https://clear.ml/docs/) in under 2 minutes**

---
**Friendly tutorials to get you started**

<table>
<tbody>
  <tr>
    <td><a href="https://github.com/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_1_Experiment_Management.ipynb"><b>Step 1</b></a> - Experiment Management</td>
    <td><a target="_blank" href="https://colab.research.google.com/github/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_1_Experiment_Management.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_2_Setting_Up_Agent.ipynb"><b>Step 2</b></a> - Remote Execution Agent Setup</td>
    <td><a target="_blank" href="https://colab.research.google.com/github/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_2_Setting_Up_Agent.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_3_Remote_Execution.ipynb"><b>Step 3</b></a> - Remotely Execute Tasks</td>
    <td><a target="_blank" href="https://colab.research.google.com/github/allegroai/clearml/blob/master/docs/tutorials/Getting_Started_3_Remote_Execution.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></td>
  </tr>
</tbody>
</table>

</div>

---

<table>
<tbody>
  <tr>
    <td>Experiment Management</td>
    <td>Datasets</td>
  </tr>
  <tr>
    <td><a href="https://app.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/experiment_manager.gif?raw=true" width="100%"></a></td>
    <td><a href="https://app.clear.ml/datasets"><img src="https://github.com/allegroai/clearml/blob/master/docs/datasets.gif?raw=true" width="100%"></a></td>
  </tr>
  <tr>
    <td colspan="2" height="24px"></td>
  </tr>
  <tr>
    <td>Orchestration</td>
    <td>Pipelines</td>
  </tr>
  <tr>
    <td><a href="https://app.clear.ml/workers-and-queues/autoscalers"><img src="https://github.com/allegroai/clearml/blob/master/docs/orchestration.gif?raw=true" width="100%"></a></td>
    <td><a href="https://app.clear.ml/pipelines"><img src="https://github.com/allegroai/clearml/blob/master/docs/pipelines.gif?raw=true" width="100%"></a></td>
  </tr>
</tbody>
</table>


## ClearML Experiment Manager

**Adding only 2 lines to your code gets you the following**

* Complete experiment setup log
    * Full source control info, including non-committed local changes
    * Execution environment (including specific packages & versions)
    * Hyper-parameters
        * [`argparse`](https://docs.python.org/3/library/argparse.html)/[Click](https://github.com/pallets/click/)/[PythonFire](https://github.com/google/python-fire) for command line parameters with currently used values
        * Explicit parameters dictionary
        * Tensorflow Defines (absl-py)
        * [Hydra](https://github.com/facebookresearch/hydra) configuration and overrides
    * Initial model weights file
* Full experiment output automatic capture
    * stdout and stderr
    * Resource Monitoring (CPU/GPU utilization, temperature, IO, network, etc.)
    * Model snapshots (With optional automatic upload to central storage: Shared folder, S3, GS, Azure, Http)
    * Artifacts log & store (Shared folder, S3, GS, Azure, Http)
    * Tensorboard/[TensorboardX](https://github.com/allegroai/clearml/tree/master/examples/frameworks/tensorboardx) scalars, metrics, histograms, **images, audio and video samples**
    * [Matplotlib & Seaborn](https://github.com/allegroai/clearml/tree/master/examples/frameworks/matplotlib)
    * [ClearML Logger](https://clear.ml/docs/latest/docs/fundamentals/logger) interface for complete flexibility.
* Extensive platform support and integrations
    * Supported ML/DL frameworks: [PyTorch](https://github.com/allegroai/clearml/tree/master/examples/frameworks/pytorch) (incl' [ignite](https://github.com/allegroai/clearml/tree/master/examples/frameworks/ignite) / [lightning](https://github.com/allegroai/clearml/tree/master/examples/frameworks/pytorch-lightning)), [Tensorflow](https://github.com/allegroai/clearml/tree/master/examples/frameworks/tensorflow), [Keras](https://github.com/allegroai/clearml/tree/master/examples/frameworks/keras), [AutoKeras](https://github.com/allegroai/clearml/tree/master/examples/frameworks/autokeras), [FastAI](https://github.com/allegroai/clearml/tree/master/examples/frameworks/fastai), [XGBoost](https://github.com/allegroai/clearml/tree/master/examples/frameworks/xgboost), [LightGBM](https://github.com/allegroai/clearml/tree/master/examples/frameworks/lightgbm), [MegEngine](https://github.com/allegroai/clearml/tree/master/examples/frameworks/megengine) and [Scikit-Learn](https://github.com/allegroai/clearml/tree/master/examples/frameworks/scikit-learn)
    * Seamless integration (including version control) with [**Jupyter Notebook**](https://jupyter.org/)
    and [*PyCharm* remote debugging](https://github.com/allegroai/trains-pycharm-plugin)
      
#### [Start using ClearML](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps) 


1. Sign up for free to the [ClearML Hosted Service](https://app.clear.ml) (alternatively, you can set up your own server, see [here](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server)).

    > **_ClearML Demo Server:_** ClearML no longer uses the demo server by default. To enable the demo server, set the `CLEARML_NO_DEFAULT_SERVER=0`
    > environment variable. Credentials aren't needed, but experiments launched to the demo server are public, so make sure not 
    > to launch sensitive experiments if using the demo server.

1. Install the `clearml` python package:

    ```bash
    pip install clearml
    ```

1. Connect the ClearML SDK to the server by [creating credentials](https://app.clear.ml/settings/workspace-configuration), then execute the command
below and follow the instructions: 

    ```bash
    clearml-init
    ```

1. Add two lines to your code:
    ```python
    from clearml import Task
    task = Task.init(project_name='examples', task_name='hello world')
    ```

And you are done! Everything your process outputs is now automagically logged into ClearML.

Next step, automation! **Learn more about ClearML's two-click automation [here](https://clear.ml/docs/latest/docs/getting_started/mlops/mlops_first_steps)**. 

## ClearML Architecture

The ClearML run-time components:

* The ClearML Python Package - for integrating ClearML into your existing scripts by adding just two lines of code, and optionally extending your experiments and other workflows with ClearML's powerful and versatile set of classes and methods.
* The ClearML Server - for storing experiment, model, and workflow data; supporting the Web UI experiment manager and MLOps automation for reproducibility and tuning. It is available as a hosted service and open source for you to deploy your own ClearML Server.
* The ClearML Agent - for MLOps orchestration, experiment and workflow reproducibility, and scalability.

<img src="https://raw.githubusercontent.com/allegroai/clearml-docs/main/docs/img/clearml_architecture.png" width="100%" alt="clearml-architecture">

## Additional Modules 

- [clearml-session](https://github.com/allegroai/clearml-session) - **Launch remote JupyterLab / VSCode-server inside any docker, on Cloud/On-Prem machines**
- [clearml-task](https://github.com/allegroai/clearml/blob/master/docs/clearml-task.md) - Run any codebase on remote machines with full remote logging of Tensorboard, Matplotlib & Console outputs 
- [clearml-data](https://github.com/allegroai/clearml/blob/master/docs/datasets.md) - **CLI for managing and versioning your datasets, including creating / uploading / downloading of data from S3/GS/Azure/NAS** 
- [AWS Auto-Scaler](https://clear.ml/docs/latest/docs/guides/services/aws_autoscaler) - Automatically spin EC2 instances based on your workloads with preconfigured budget! No need for AKE!
- [Hyper-Parameter Optimization](https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt) - Optimize any code with black-box approach and state-of-the-art Bayesian optimization algorithms
- [Automation Pipeline](https://clear.ml/docs/latest/docs/guides/pipeline/pipeline_controller) - Build pipelines based on existing experiments / jobs, supports building pipelines of pipelines!  
- [Slack Integration](https://clear.ml/docs/latest/docs/guides/services/slack_alerts) - Report experiments progress / failure directly to Slack (fully customizable!)  

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
  - Store all your data on any object-storage solution, with the most straightforward interface possible
  - Make your data transparent by cataloging it all on the ClearML platform

We believe ClearML is ground-breaking. We wish to establish new standards of true seamless integration between
experiment management, MLOps, and data management.

## Who We Are

ClearML is supported by you and the [clear.ml](https://clear.ml) team, which helps enterprise companies build scalable MLOps. 

We built ClearML to track and control the glorious but messy process of training production-grade deep learning models.
We are committed to vigorously supporting and expanding the capabilities of ClearML.

We promise to always be backwardly compatible, making sure all your logs, data, and pipelines will always upgrade with you.

## License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)

If ClearML is part of your development process / project / publication, please cite us :heart: : 
```
@misc{clearml,
title = {ClearML - Your entire MLOps stack in one open-source tool},
year = {2023},
note = {Software available from http://github.com/allegroai/clearml},
url={https://clear.ml/},
author = {ClearML},
}
```

## Documentation, Community & Support

For more information, see the [official documentation](https://clear.ml/docs) and [on YouTube](https://www.youtube.com/c/ClearML).

For examples and use cases, check the [examples folder](https://github.com/allegroai/clearml/tree/master/examples) and [corresponding documentation](https://clear.ml/docs/latest/docs/guides).

If you have any questions: post on our [Slack Channel](https://joinslack.clear.ml), or tag your questions on [stackoverflow](https://stackoverflow.com/questions/tagged/clearml) with '**[clearml](https://stackoverflow.com/questions/tagged/clearml)**' tag (*previously [trains](https://stackoverflow.com/questions/tagged/trains) tag*).

For feature requests or bug reports, please use [GitHub issues](https://github.com/allegroai/clearml/issues).

Additionally, you can always find us at *info@clear.ml*

## Contributing

**PRs are always welcome** :heart: See more details in the ClearML [Guidelines for Contributing](https://github.com/allegroai/clearml/blob/master/docs/contributing.md).


_May the force (and the goddess of learning rates) be with you!_
