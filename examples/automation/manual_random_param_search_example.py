from random import sample

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='Random Hyper-Parameter Search Example', task_type=Task.TaskTypes.optimizer)

# Create a hyper-parameter dictionary for the task
params = dict()

# track my parameters dictionary
params = task.connect(params)

# define random search space,
params['batch_size'] = [64, 96, 128, 160, 192]
params['layer_1'] = [128, 512, 32]
params['layer_2'] = [128, 512, 32]

# This is a simple random search
# (can be integrated with 'bayesian-optimization' 'hpbandster' etc.)
space = {
    'batch_size': lambda: sample(params['batch_size'], 1)[0],
    'layer_1': lambda: sample(range(*params['layer_1']), 1)[0],
    'layer_2': lambda: sample(range(*params['layer_2']), 1)[0],
}

# number of random samples to test from 'space'
params['total_number_of_experiments'] = 3

# execution queue to add experiments to
params['execution_queue_name'] = 'default'

# experiment template to optimize with random parameter search
params['experiment_template_name'] = 'Keras HP optimization base'

# Select base template task
# Notice we can be more imaginative and use task_id which will eliminate the need to use project name
template_task = Task.get_task(project_name='examples', task_name=params['experiment_template_name'])

for i in range(params['total_number_of_experiments']):
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
    Task.enqueue(cloned_task.id, queue_name=params['execution_queue_name'])
    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

# we are done, the next step is to watch the experiments graphs
print('Done')
