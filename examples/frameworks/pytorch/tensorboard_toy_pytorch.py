import os
from tempfile import gettempdir

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='pytorch tensorboard toy example')

writer = SummaryWriter(log_dir=os.path.join(gettempdir(), 'tensorboard_logs'))

# convert to 4d [batch, col, row, RGB-channels]
image_open = Image.open(os.path.join("..", "..", "reporting", "data_samples", "picasso.jpg"))
image = np.asarray(image_open)
image_gray = image[:, :, 0][np.newaxis, :, :, np.newaxis]
image_rgba = np.concatenate((image, 255*np.atleast_3d(np.ones(shape=image.shape[:2], dtype=np.uint8))), axis=2)
image_rgba = image_rgba[np.newaxis, :, :, :]
image = image[np.newaxis, :, :, :]

writer.add_image("test/first", image[0], dataformats='HWC')
writer.add_image("test_gray/second", image_gray[0], dataformats='HWC')
writer.add_image("test_rgba/third", image_rgba[0], dataformats='HWC')
# writer.add_image("image/first_series", image, max_outputs=10)
# writer.add_image("image_gray/second_series", image_gray, max_outputs=10)
# writer.add_image("image_rgba/third_series", image_rgba, max_outputs=10)

print('Done!')
