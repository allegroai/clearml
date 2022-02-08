# ClearML - Example of tensorflow eager mode, model logging and tensorboard
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A deep MNIST classifier using convolutional layers.
Sample usage:
  python mnist.py --help
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time
from tempfile import gettempdir

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from clearml import Task


tf.compat.v1.enable_eager_execution()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='TensorFlow eager mode')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 100, """Flag of type integer""")
tf.app.flags.DEFINE_string('img_path', './img', """Flag of type string""")


layers = tf.keras.layers
FLAGS = None


class Discriminator(tf.keras.Model):
    """
    GAN Discriminator.
    A network to differentiate between generated and real handwritten digits.
    """

    def __init__(self, data_format):
        """Creates a model for discriminating between real and generated digits.
        Args:
          data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        super(Discriminator, self).__init__(name='')
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 28, 28, 1]
        self.conv1 = layers.Conv2D(
            64, 5, padding='SAME', data_format=data_format, activation=tf.tanh)
        self.pool1 = layers.AveragePooling2D(2, 2, data_format=data_format)
        self.conv2 = layers.Conv2D(
            128, 5, data_format=data_format, activation=tf.tanh)
        self.pool2 = layers.AveragePooling2D(2, 2, data_format=data_format)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation=tf.tanh)
        self.fc2 = layers.Dense(1, activation=None)

    def call(self, inputs):
        """Return two logits per image estimating input authenticity.
        Users should invoke __call__ to run the network, which delegates to this
        method (and not call this method directly).
        Args:
          inputs: A batch of images as a Tensor with shape [batch_size, 28, 28, 1]
            or [batch_size, 1, 28, 28]
        Returns:
          A Tensor with shape [batch_size] containing logits estimating
          the probability that corresponding digit is real.
        """
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Generator(tf.keras.Model):
    """
    Generator of handwritten digits similar to the ones in the MNIST dataset.
    """

    def __init__(self, data_format):
        """Creates a model for discriminating between real and generated digits.
        Args:
          data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        super(Generator, self).__init__(name='')
        self.data_format = data_format
        # We are using 128 6x6 channels as input to the first deconvolution layer
        if data_format == 'channels_first':
            self._pre_conv_shape = [-1, 128, 6, 6]
        else:
            assert data_format == 'channels_last'
            self._pre_conv_shape = [-1, 6, 6, 128]
        self.fc1 = layers.Dense(6 * 6 * 128, activation=tf.tanh)

        # In call(), we reshape the output of fc1 to _pre_conv_shape

        # Deconvolution layer. Resulting image shape: (batch, 14, 14, 64)
        self.conv1 = layers.Conv2DTranspose(
            64, 4, strides=2, activation=None, data_format=data_format)

        # Deconvolution layer. Resulting image shape: (batch, 28, 28, 1)
        self.conv2 = layers.Conv2DTranspose(
            1, 2, strides=2, activation=tf.nn.sigmoid, data_format=data_format)

    def call(self, inputs):
        """Return a batch of generated images.
        Users should invoke __call__ to run the network, which delegates to this
        method (and not call this method directly).
        Args:
          inputs: A batch of noise vectors as a Tensor with shape
            [batch_size, length of noise vectors].
        Returns:
          A Tensor containing generated images. If data_format is 'channels_last',
          the shape of returned images is [batch_size, 28, 28, 1], else
          [batch_size, 1, 28, 28]
        """

        x = self.fc1(inputs)
        x = tf.reshape(x, shape=self._pre_conv_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def discriminator_loss(discriminator_real_outputs, discriminator_gen_outputs):
    """
    Original discriminator loss for GANs, with label smoothing.
    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
    details.
    Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    Returns:
    A scalar loss Tensor.
    """

    loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
      tf.ones_like(discriminator_real_outputs),
      discriminator_real_outputs,
      label_smoothing=0.25)
    loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
      tf.zeros_like(discriminator_gen_outputs), discriminator_gen_outputs)
    loss = loss_on_real + loss_on_generated
    tf.contrib.summary.scalar('discriminator_loss', loss)
    return loss


def generator_loss(discriminator_gen_outputs):
    """
    Original generator loss for GANs.
    L = -log(sigmoid(D(G(z))))
    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661)
    for more details.
    Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    Returns:
    A scalar loss Tensor.
    """
    loss = tf.compat.v1.losses.sigmoid_cross_entropy(
      tf.ones_like(discriminator_gen_outputs), discriminator_gen_outputs)
    tf.contrib.summary.scalar('generator_loss', loss)
    return loss


def train_one_epoch(generator, discriminator, generator_optimizer,
                    discriminator_optimizer, dataset, step_counter,
                    log_interval, noise_dim):
    """
    Train `generator` and `discriminator` models on `dataset`.
    Args:
    generator: Generator model.
    discriminator: Discriminator model.
    generator_optimizer: Optimizer to use for generator.
    discriminator_optimizer: Optimizer to use for discriminator.
    dataset: Dataset of images to train on.
    step_counter: An integer variable, used to write summaries regularly.
    log_interval: How many steps to wait between logging and collecting
      summaries.
    noise_dim: Dimension of noise vector to use.
    """

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    for (batch_index, images) in enumerate(dataset):
        with tf.device('/cpu:0'):
            tf.compat.v1.assign_add(step_counter, 1)
        with tf.contrib.summary.record_summaries_every_n_global_steps(
                log_interval, global_step=step_counter):
            current_batch_size = images.shape[0]
            noise = tf.random.uniform(
                  shape=[current_batch_size, noise_dim],
                  minval=-1.,
                  maxval=1.,
                  seed=batch_index)

            # we can use 2 tapes or a single persistent tape.
            # Using two tapes is memory efficient since intermediate tensors can be
            # released between the two .gradient() calls below
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise)
                tf.contrib.summary.image(
                    'generated_images',
                    tf.reshape(generated_images, [-1, 28, 28, 1]),
                    max_images=10)

                discriminator_gen_outputs = discriminator(generated_images)
                discriminator_real_outputs = discriminator(images)
                discriminator_loss_val = discriminator_loss(discriminator_real_outputs,
                                                            discriminator_gen_outputs)
                total_discriminator_loss += discriminator_loss_val

                generator_loss_val = generator_loss(discriminator_gen_outputs)
                total_generator_loss += generator_loss_val

            generator_grad = gen_tape.gradient(generator_loss_val,
                                               generator.variables)
            discriminator_grad = disc_tape.gradient(discriminator_loss_val,
                                                    discriminator.variables)

            generator_optimizer.apply_gradients(
                zip(generator_grad, generator.variables))
            discriminator_optimizer.apply_gradients(
                zip(discriminator_grad, discriminator.variables))

            if log_interval and batch_index > 0 and batch_index % log_interval == 0:
                print('Batch #%d\tAverage Generator Loss: %.6f\tAverage Discriminator Loss: %.6f' %
                      (batch_index, total_generator_loss / batch_index, total_discriminator_loss / batch_index))


def main(_):
    (device, data_format) = ('/gpu:0', 'channels_first')
    if FLAGS.no_gpu or tf.contrib.eager.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    data = input_data.read_data_sets(FLAGS.data_dir)
    dataset = (
          tf.data.Dataset.from_tensor_slices(data.train.images[:1280]).shuffle(60000).batch(FLAGS.batch_size))

    # Create the models and optimizers.
    model_objects = {
        'generator': Generator(data_format),
        'discriminator': Discriminator(data_format),
        'generator_optimizer': tf.compat.v1.train.AdamOptimizer(FLAGS.lr),
        'discriminator_optimizer': tf.compat.v1.train.AdamOptimizer(FLAGS.lr),
        'step_counter': tf.compat.v1.train.get_or_create_global_step(),
    }

    # Prepare summary writer and checkpoint info
    summary_writer = tf.contrib.summary.create_file_writer(
          FLAGS.output_dir, flush_millis=1000)
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
    latest_cpkt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if latest_cpkt:
        print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint = tf.train.Checkpoint(**model_objects)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)

    with tf.device(device):
        for _ in range(3):
            start = time.time()
            with summary_writer.as_default():
                train_one_epoch(dataset=dataset, log_interval=FLAGS.log_interval,
                                noise_dim=FLAGS.noise, **model_objects)
            end = time.time()
            checkpoint.save(checkpoint_prefix)
            print('\nTrain time for epoch #%d (step %d): %f' %
                  (checkpoint.save_counter.numpy(), checkpoint.step_counter.numpy(), end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data-dir',
            type=str,
            default=os.path.join(gettempdir(), 'tensorflow', 'mnist', 'input_data'),
            help='Directory for storing input data (default /tmp/tensorflow/mnist/input_data)')
    parser.add_argument(
            '--batch-size',
            type=int,
            default=16,
            metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument(
            '--log-interval',
            type=int,
            default=1,
            metavar='N',
            help='number of batches between logging and writing summaries (default: 100)')
    parser.add_argument(
            '--output_dir',
            type=str,
            default=os.path.join(gettempdir(), 'tensorflow'),
            metavar='DIR',
            help='Directory to write TensorBoard summaries (defaults to none)')
    parser.add_argument(
            '--checkpoint_dir',
            type=str,
            default=os.path.join(gettempdir(), 'tensorflow', 'mnist', 'checkpoints'),
            metavar='DIR',
            help='Directory to save checkpoints in (once per epoch) (default /tmp/tensorflow/mnist/checkpoints/)')
    parser.add_argument(
            '--lr',
            type=float,
            default=0.001,
            metavar='LR',
            help='learning rate (default: 0.001)')
    parser.add_argument(
            '--noise',
            type=int,
            default=100,
            metavar='N',
            help='Length of noise vector for generator input (default: 100)')
    parser.add_argument(
            '--no-gpu',
            action='store_true',
            default=False,
            help='disables GPU usage even if a GPU is available')

    FLAGS, unparsed = parser.parse_known_args()

tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
