from clearml import Task
from time import sleep

# Initialize the Task Pipe's first Task used to start the Task Pipe
task = Task.init('examples', 'Simple Controller Task', task_type=Task.TaskTypes.controller)

# Create a hyper-parameter dictionary for the task
param = dict()
# Connect the hyper-parameter dictionary to the task
param = task.connect(param)

# In this example we pass next task's name as a parameter
param['next_task_name'] = 'Toy Base Task'
# This is a parameter name in the next task we want to change
param['param_name'] = 'Example_Param'
# This is the parameter value in the next task we want to change
param['param_name_new_value'] = 3
# The queue where we want the template task (clone) to be sent to
param['execution_queue_name'] = 'default'

# Simulate the work of a Task
print('Processing....')
sleep(2.0)
print('Done processing :)')

# Get a reference to the task to pipe to.
next_task = Task.get_task(project_name=task.get_project_name(), task_name=param['next_task_name'])

# Clone the task to pipe to. This creates a task with status Draft whose parameters can be modified.
cloned_task = Task.clone(source_task=next_task, name='Auto generated cloned task')

# Get the original parameters of the Task, modify the value of one parameter,
#   and set the parameters in the next Task
cloned_task_parameters = cloned_task.get_parameters()
cloned_task_parameters[param['param_name']] = param['param_name_new_value']
cloned_task.set_parameters(cloned_task_parameters)

# Enqueue the Task for execution. The enqueued Task must already exist in the clearml platform
print('Enqueue next step in pipeline to queue: {}'.format(param['execution_queue_name']))
Task.enqueue(cloned_task.id, queue_name=param['execution_queue_name'])

# We are done. The next step in the pipe line is in charge of the pipeline now.
print('Done')
