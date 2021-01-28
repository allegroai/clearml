# This Task is the base task that we will be executing as a second step (see task_piping.py)
# In order to make sure this experiment is registered in the platform, you must execute it once.

from clearml import Task

# Initialize the task pipe's first task used to start the task pipe
task = Task.init('examples', 'Toy Base Task')

# Create a dictionary for hyper-parameters
params = dict()

# Add a parameter and value to the dictionary
params['Example_Param'] = 1

# Connect the hyper-parameter dictionary to the task
task.connect(params)

# Print the value to demonstrate it is the value is set by the initiating task.
print("Example_Param is {}".format(params['Example_Param']))
