from trains import Task
from trains.automation.controller import PipelineController


task = Task.init(project_name='examples', task_name='pipeline demo', task_type=Task.TaskTypes.controller)

pipe = PipelineController(default_execution_queue='default')
pipe.add_step(name='stage_data', base_task_project='examples', base_task_name='pipeline step 1 dataset artifact')
pipe.add_step(name='stage_process', parents=['stage_data', ],
              base_task_project='examples', base_task_name='pipeline step 2 process dataset',
              parameter_override={'General/dataset_url': '${stage_data.artifacts.dataset.url}',
                                  'General/test_size': '0.25'})
pipe.add_step(name='stage_train', parents=['stage_process', ],
              base_task_project='examples', base_task_name='pipeline step 3 train model',
              parameter_override={'General/dataset_task_id': '${stage_process.id}'})

# Starting the pipeline (in the background)
pipe.start()
# Wait until pipeline terminates
pipe.wait()
# cleanup everything
pipe.stop()

print('done')
