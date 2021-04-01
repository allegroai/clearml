from clearml import Task


for i in range(3):
    task = Task.init(project_name="same process, multiple tasks", task_name="Task #{}".format(i))
    # Doing Task processing here
    #
    #
    task.close()
