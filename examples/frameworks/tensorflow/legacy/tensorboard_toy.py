# ClearML - Example of tensorboard with tensorflow (without any actual training)
#
import os
from tempfile import gettempdir

import tensorflow as tf
import numpy as np
from PIL import Image

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='tensorboard toy example')

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)
tf.summary.scalar("normal/value", mean_moving_normal[-1])

# Make a normal distribution with shrinking variance
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)
tf.summary.scalar("normal/variance_shrinking_normal", variance_shrinking_normal[-1])

# Let's combine both of those distributions into one dataset
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)
tf.summary.scalar("normal/normal_combined", normal_combined[0])

# Add a gamma distribution
gamma = tf.random_gamma(shape=[1000], alpha=k)
tf.summary.histogram("gamma", gamma)

# And a poisson distribution
poisson = tf.random_poisson(shape=[1000], lam=k)
tf.summary.histogram("poisson", poisson)

# And a uniform distribution
uniform = tf.random_uniform(shape=[1000], maxval=k*10)
tf.summary.histogram("uniform", uniform)

# Finally, combine everything together!
all_distributions = [mean_moving_normal, variance_shrinking_normal, gamma, poisson, uniform]
all_combined = tf.concat(all_distributions, 0)
tf.summary.histogram("all_combined", all_combined)

# Log text value
tf.summary.text("this is a test", tf.make_tensor_proto("This is the content", dtype=tf.string))

# convert to 4d [batch, col, row, RGB-channels]
image_open = Image.open(os.path.join("..", "..", "..", "reporting", "data_samples", "picasso.jpg"))
image = np.asarray(image_open)
image_gray = image[:, :, 0][np.newaxis, :, :, np.newaxis]
image_rgba = np.concatenate((image, 255*np.atleast_3d(np.ones(shape=image.shape[:2], dtype=np.uint8))), axis=2)
image_rgba = image_rgba[np.newaxis, :, :, :]
image = image[np.newaxis, :, :, :]

tf.summary.image("test", image, max_outputs=10)
tf.summary.image("test_gray", image_gray, max_outputs=10)
tf.summary.image("test_rgba", image_rgba, max_outputs=10)

# Setup a session and summary writer
summaries = tf.summary.merge_all()
sess = tf.Session()

logger = task.get_logger()

# Use original FileWriter for comparison , run:
# % tensorboard --logdir=/tmp/histogram_example
writer = tf.summary.FileWriter(os.path.join(gettempdir(), "histogram_example"))

# Setup a loop and write the summaries to disk
N = 40
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)

print('Done!')
