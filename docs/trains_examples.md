# TRAINS Usage Examples

## Introduction
TRAINS includes usage examples for the *Keras*, *PyTorch*, and *TensorFlow* deep learning frameworks, 
as well as *Jupyter Notebook* integration and custom examples for reporting metrics, configuring models.
You can run these examples and view their results on the TRAINS Web-App.

The examples are described below, including a link for the source code
and expected results for each run. 
* For each example, only two lines of TRAINS integration code, were added

        from trains import Task
        task = Task.init(project_name=”examples”, task_name=”description”)

## Viewing experiment results

In order to view an experiment's results (or other details) you can either:

1. Open the TRAINS Web-App in your browser and login.
2. On the Home page, in the *recent project* section, click the card for the project containing the experiment 
(example experiments can be found under the *examples* project card).
3. In the *Experiments* tab, click your experiment. The details panel slides open.
4. Choose the experiment details by clicking one of the information tabs.

OR

1. While running the experiment, a direct link for a dedicated results page is printed. 


# Keras Examples

### Keras with TensorBoard - MNIST Training

[keras_tensorboard.py](https://github.com/allegroai/trains/blob/master/examples/keras_tensorboard.py)
is an example of training a small convolutional NN on the MNIST DataSet.

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Input model weights, if executed for the second time (loaded from the previous checkpoint)
    * Input model’s creator experiment (a link to the experiment details in the *EXPERIMENTS* page)
    * Output model + Configuration 
* **RESULTS**
    * **SCALARS**: Accuracy/loss scalar metric graphs
    * **PLOTS**: Convolution weights histograms
    * **LOG**: Console standard output/error

# Pytorch Examples

### PyTorch - MNIST Training

[pytorch_mnist.py](https://github.com/allegroai/trains/blob/master/examples/pytorch_mnist.py) is an example
of PyTorch MNIST training integration.

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Input model weights, if executed for the second time (loaded from the previous checkpoint)
    * Input model’s creator experiment (a link to the experiment details in the *EXPERIMENTS* page)
    * Output model (a link to the output model details in the *MODELS* page)
* **RESULTS**
    * **LOG**: Console standard output/error

### PyTorch and Matplotlib - Testing Style Transfer

[pytorch_matplotlib.py](https://github.com/allegroai/trains/blob/master/examples/pytorch_matplotlib.py)
is an example of
connecting the neural style transfer from the official PyTorch tutorial to TRAINS.
Neural-Style, or Neural-Transfer, allows you to take an image and
reproduce it with a new artistic style. The algorithm takes three images 
(an input image, a content-image, and a style-image) and change the input
to resemble the content of the content-image and the artistic style of the style-image.

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Input model (a link to the input model details in the *MODELS* page)
    * Output model (a link to the output model details in the *MODELS* page)
* **RESULTS**
    * **DEBUG IMAGES**: Input image, input style images, an output transferred style image
    * **LOG**: Console standard output/error

### PyTorch with Tensorboard - MNIST Train

[pytorch_tensorboard.py](https://github.com/allegroai/trains/blob/master/examples/pytorch_tensorboard.py)
is an example of PyTorch MNIST training running with Tensorboard

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Input model, if executed for the second time (a link to the input model details in the *MODELS* page)
    * Input model’s creator experiment (a link to the experiment details in the *EXPERIMENTS* page)
    * Output model (a link to the output model details in the *MODELS* page)
* **RESULTS**
    * **SCALARS**: Train and test loss scalars
    * **LOG**: Console standard output/error

### PyTorch with tensorboardX - MNIST Train

[pytorch_tensorboardX.py](https://github.com/allegroai/trains/blob/master/examples/pytorch_tensorboardX.py)
is an example of PyTorch MNIST training running with tensorboardX

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Input model, if executed for the second time (a link to the input model details in the *MODELS* page)
    * Input model’s creator experiment (a link to the experiment details in the *EXPERIMENTS* page)
    * Output model (a link to the output model details in the *MODELS* page)
* **RESULTS**
    * **SCALARS**: Train and test loss scalars
    * **LOG**: Console standard output/error

# TensorFlow Examples

### TensorBoard with TensorFlow (without Training)

[tensorboard_toy.py](https://github.com/allegroai/trains/blob/master/examples/tensorboard_toy.py)
is a toy example of TensorBoard.

**View Example Output**

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **RESULTS**
    * **SCALARS**: Random variable samples scalars
    * **PLOTS**: Random variable samples histograms
    * **DEBUG IMAGES**: Test images
    * **LOG**: Console standard output/error

### TensorFlow in Eager Mode 

[tensorflow_eager.py](https://github.com/allegroai/trains/blob/master/examples/tensorflow_eager.py)
is an example of running Tensorflow in eager mode 

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **RESULTS**
    * **SCALARS**: Generator and discriminator loss
    * **DEBUG IMAGES**: Generated images
    * **LOG**: Console standard output/error

### TensorBoard Plugin - Precision Recall Curves

[tensorboard_pr_curve.py](https://github.com/allegroai/trains/blob/master/examples/tensorboard_pr_curve.py)
is an example of TensorBoard precision recall curves

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **RESULTS**
    * **PLOTS**: Precision recall curves
    * **DEBUG IMAGES**: Generated images
    * **LOG**: Console standard output/error

### Tensorflow Flags / absl
##### Toy Tensorflow FLAGS logging with absl

[absl_example.py](https://github.com/allegroai/trains/blob/master/examples/absl_example.py) 
is an example of toy Tensorflow FLAGS logging with absl package (*absl-py*) coupled with parameters dictionary

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Tensorflow flags (with 'TF_DEFINE/' prefix)
* **RESULTS**
    * **LOG**: Console standard output/error

### TensorFlow MNIST Classifier with TensorBoard Reports

[tensorflow_mnist_with_summaries.py](https://github.com/allegroai/trains/blob/master/examples/tensorflow_mnist_with_summaries.py)
is an example of Tensorflow MNIST with TensorBoard summary, model storage, and logging.

Relevant outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Command line arguments
* **MODEL**
    * Output model (a link to the output model details in the *MODELS* page)
* **RESULTS**
    * **SCALARS**: Network statistics across the training steps (e.g., cross entropy, dropout, and specific layer statistics) 
    * **PLOTS**: Convolutional layer histogram
    * **DEBUG IMAGES**: Sample of the network input images
    * **LOG**: Console standard output/error


# *Jupyter Notebook* Example

[jupyter.ipynb](https://github.com/allegroai/trains/blob/master/examples/jupyter.ipynb)
is an example of integrating matplotlib and training with keras on
*Jupyter Notebook*. 
This example connects a parameters dictionary, prints simple graphs and trains an MNIST classifier using Keras.
 
Relevant Outputs

* **EXECUTION**
    * **HYPER PARAMETERS**: Parameter dictionary
* **MODEL**
    * Output model (a link to the output model details in the *MODELS* page)
    * Model Configuration
* **RESULTS**
    * **SCALARS**: Training loss across iterations
    * **PLOTS**: Sine and circles plots, convolution weights histogram
    * **LOG**: Console standard output/error
    
    
# Custom Examples

### Manual Reporting

[manual_reporting.py](https://github.com/allegroai/trains/blob/master/examples/manual_reporting.py)
is an example of manually reporting graphs and statistics.

Relevant outputs

* **RESULTS**
    * **SCALARS**: Scalar graphs
    * **PLOTS**: Confusion matrix, histogram, 2D scatter plot, 3D scatter plot
    * **DEBUG IMAGES**: Uploaded example images
    * **LOG**: Console standard output/error
    
### Manual Model Configuration

[manual_model_config.py](https://github.com/allegroai/trains/blob/master/examples/manual_model_config.py)
is an example of manually configuring a model, model storage, label enumeration values, and logging.

Relevant Outputs

* **MODEL**
    * Output model (a link to the output model details in the *MODELS* page, including **label enumeration** values)
    * Model Configuration 
* **RESULTS**
    * **LOG**: Console standard output/error
