from clearml import Task


for i in range(3):
    task = Task.init(project_name="examples", task_name="Same process, Multiple tasks, Task #{}".format(i))
    # Doing Task processing here
    print("Task #{} running".format(i))
    #
    print("Task #{} done :) ".format(i))
    task.close()
