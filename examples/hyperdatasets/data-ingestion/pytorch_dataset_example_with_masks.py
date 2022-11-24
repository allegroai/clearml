import numpy as np
import torch.utils.data
from allegroai import DataView, FrameGroup, Task
from PIL import Image
from torch.utils.data import DataLoader


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, dv):
        # automatically adjust dataset to balance all queries
        self.frames = dv.to_list()

    def __getitem__(self, idx):
        frame_group = self.frames[idx]  # type: FrameGroup
        img_path = frame_group["000002"].get_local_source()
        img = Image.open(img_path).convert("RGB").resize((256, 256))

        mask_path = frame_group["000002"].get_local_mask_source()
        mask = Image.open(mask_path).resize((256, 256))

        return np.array(img), np.array(mask),

    def __len__(self):
        return len(self.frames)


task = Task.init(project_name='examples', task_name='PyTorch Sample Dataset with Masks')

# Create DataView with example query
dataview = DataView()
dataview.add_query(dataset_name='sample-dataset-masks', version_name='Current')
# dataview.add_query(dataset_name='sample-dataset', version_name='Current', roi_query=["aeroplane"])

# if we want all files to be downloaded in the background, we can call prefetch
# dataview.prefetch_files()

# create PyTorch Dataset
dataset = ExampleDataset(dataview)

# do your thing here :)
print('Fake PyTorch stuff below:')
print('Dataset length', len(dataset))

torch.manual_seed(0)
data_loader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=1,
    pin_memory=True,
    prefetch_factor=2,
)
for i, data in enumerate(data_loader):
    print('{}] {}'.format(i, data))

print('done')
