from clearml import Task
from clearml.automation import TaskScheduler


def simple_function():
    print('This code is executed in a background thread, '
          'on the same machine as the TaskScheduler process')
    # add some logic here
    print('done')


# Create the scheduler controller
scheduler = TaskScheduler()

# Run the Task 'PyTorch MNIST train', every day at 10:30am
scheduler.add_task(
    name='recurring pipeline job',
    schedule_task_id=Task.get_task(project_name='examples', task_name='PyTorch MNIST train'),
    queue='default',
    minute=30,
    hour=10,
    day=1,
    recurring=True,
)


# a few more Examples:
#         Launch every 15 minutes
# scheduler.add_task(schedule_task_id='1235', queue='default', minute=15)
#         Launch every 1 hour
# scheduler.add_task(schedule_task_id='1235', queue='default', hour=1)
#         Launch every 1 hour at hour:30 minutes (i.e. 1:30, 2:30 etc.)
# scheduler.add_task(schedule_task_id='1235', queue='default', hour=1, minute=30)
#         Launch every day at 22:30 (10:30 pm)
# scheduler.add_task(schedule_task_id='1235', queue='default', minute=30, hour=22, day=1)
#         Launch every other day at 7:30 (7:30 am)
# scheduler.add_task(schedule_task_id='1235', queue='default', minute=30, hour=7, day=2)
#         Launch every Saturday at 8:30am (notice `day=0`)
# scheduler.add_task(schedule_task_id='1235', queue='default', minute=30, hour=8, day=0, weekdays=['saturday'])
#         Launch every 2 hours on the weekends Saturday/Sunday (notice `day` is not passed)
# scheduler.add_task(schedule_task_id='1235', queue='default', hour=2, weekdays=['saturday', 'sunday'])
#         Launch once a month at the 5th of each month
# scheduler.add_task(schedule_task_id='1235', queue='default', month=1, day=5)
#         Launch once a year on March 4th of each year
# scheduler.add_task(schedule_task_id='1235', queue='default', year=1, month=3, day=4)


# Run a simple logic function, every 2 hours, at minute 30 of every hour, only on business days (mon-fri)
scheduler.add_task(
    name='workdays mock job',
    schedule_function=simple_function,
    minute=30,
    hour=2,
    weekdays=['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    recurring=True,
)

#scheduler.start_remotely(queue='services')
scheduler.start()

print('This line will run remotely')
