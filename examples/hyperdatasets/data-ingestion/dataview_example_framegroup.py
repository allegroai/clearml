from allegroai import Task, DataView


task = Task.init(project_name="examples", task_name="dataview example with masks")

# simple query
dataview = DataView(iteration_order='random')
dataview.set_iteration_parameters(random_seed=123)

dataview.add_query(dataset_name='sample-dataset-masks', version_name='Current')

# print the number of frames the queries return
print("count", dataview.get_count())

# generate a list of FrameGroups from the query
# Note that the metadata is cached locally, it means the next time we call to_list() it will return faster.
list_frame_groups = dataview.to_list()

# A FrameGroup is a dictionary of SingleFrames - you can access each object with the key it was register with ("000002")
print([frame_group["000002"].get_local_source() for frame_group in list_frame_groups])

print("now in iterator form")

# iterator version of the same code, notice this time metadata is not locally cached
for frame_group in dataview:
    for key in frame_group.keys():
        print(frame_group[key].get_local_source(), frame_group[key].get_local_mask_source())

print("done")
