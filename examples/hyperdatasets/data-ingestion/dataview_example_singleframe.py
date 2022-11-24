"""
How to access and go over data
The general flow:
 - Create new dataview.
 - Query your dataview.
 - Two ways to go over the frames:
   - dataview.get_iterator()
   - dataview.to_list()
"""
from allegroai import Task, DataView


task = Task.init(project_name="examples", task_name="dataview example")

# simple query
dataview = DataView(iteration_order='random')
dataview.set_iteration_parameters(random_seed=123)

# We can query our dataset(s) with `add_query` function, for all the data use roi_query="*" or
# use only dataset and version.
# This is a general example, you can change the parameters of the `add_query` function
dataview.add_query(dataset_name='sample-dataset', version_name='Current', roi_query=["aeroplane"])

# print the number of frames the queries return
print("count", dataview.get_count())

# generate a list of FrameGroups from the query
# Note that the metadata is cached locally, it means the next time we call to_list() it will return faster.
list_single_frames = dataview.to_list()
print([f.get_local_source() for f in list_single_frames])

print("now in iterator form")

# iterator version of the same code, notice this time metadata is not locally cached
for f in dataview:
    print(f.get_local_source())

print("done")
