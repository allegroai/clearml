from random import random, sample
from trains import Task


# define random search space,
# This is a simple random search
# (can be integrated with 'bayesian-optimization' 'hpbandster' etc.)
space = {
    'batch_size': lambda: sample([64, 96, 128, 160, 192], 1)[0],
    'layer_1': lambda: sample(range(128, 512, 32), 1)[0],
    'layer_2': lambda: sample(range(128, 512, 32), 1)[0],
}

# number of random samples to test from 'space'
total_number_of_experiments = 3

# execution queue to add experiments to
execution_queue_name = 'default'

# Select base template task
# Notice we can be more imaginative and use task_id which will eliminate the need to use project name
template_task = Task.get_task(project_name='examples', task_name='Keras AutoML base')

for i in range(total_number_of_experiments):
    # clone the template task into a new write enabled task (where we can change parameters)
    cloned_task = Task.clone(source_task=template_task,
                             name=template_task.name+' {}'.format(i), parent=template_task.id)

    # get the original template parameters
    cloned_task_parameters = cloned_task.get_parameters()

    # override with random samples form grid
    for k in space.keys():
        cloned_task_parameters[k] = space[k]()

    # put back into the new cloned task
    cloned_task.set_parameters(cloned_task_parameters)
    print('Experiment {} set with parameters {}'.format(i, cloned_task_parameters))

    # enqueue the task for execution
    Task.enqueue(cloned_task.id, queue_name=execution_queue_name)
    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

# we are done, the next step is to watch the experiments graphs
print('Done')
