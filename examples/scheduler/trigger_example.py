from clearml import Task, Dataset, Model
from clearml.automation import TriggerScheduler


def trigger_model_func(model_id):
    model = Model(model_id=model_id)
    print('model id {} modified'.format(model.id))


def trigger_dataset_func(dataset_id):
    dataset = Dataset.get(dataset_id=dataset_id)
    print('dataset id {} created'.format(dataset.id))


def trigger_task_func(task_id):
    task = Task.get_task(task_id=task_id)
    print('Task ID {} metric above threshold'.format(task.id))


if __name__ == '__main__':
    # create the TriggerScheduler object (checking system state every minute)
    trigger = TriggerScheduler(pooling_frequency_minutes=1.0)

    # Add trigger on model publishing
    trigger.add_model_trigger(
        name='model deploy',
        schedule_function=trigger_model_func,
        # schedule_task_id='task_id_here', # you can also schedule an existing task to be executed
        trigger_project='examples',
        trigger_on_tags=['deploy']
    )

    # Add trigger on model publishing
    trigger.add_model_trigger(
        name='model quality check',
        # schedule_function=trigger_model_func, # you can also schedule a function to be executed.
        schedule_task_id='add_task_id_here',
        schedule_queue='default',
        trigger_project='examples',
        trigger_on_tags=['deploy']
    )

    # Add trigger on dataset creation
    trigger.add_dataset_trigger(
        name='retrain on dataset',
        schedule_function=trigger_dataset_func,
        # schedule_task_id='aabbcc', # you can also schedule an existing task to be executed
        trigger_project='datasets',
        trigger_on_tags=['retrain']
    )

    # Add trigger on Task performance
    trigger.add_task_trigger(
        name='performance high-score',
        schedule_function=trigger_task_func,
        # schedule_task_id='task_id_here', # you can also schedule an existing task to be executed
        trigger_project='examples',
        trigger_on_metric='epoch_accuracy', trigger_on_variant='epoch_accuracy',
        trigger_on_sign='max',
        trigger_on_threshold=0.99
    )

    # start the trigger daemon (locally/remotely)
    # trigger.start()
    trigger.start_remotely()
