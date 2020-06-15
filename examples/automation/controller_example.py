pipeline_node = {
    "step1": {
        # identify the node, so that we code reference outputs, use only alphanumeric characters
        "node_name": "step1",
        # parent node, to be executed before this step
        "parent_node": None,
        # the experiment/task id to clone & execute
        "base_task_id": "gafghafh",
        # preferred queue name/id to use for execution
        "queue": None,
        # preferred docker image (override experiment request)
        "docker": None,
        # parameter overrides
        "parameter_override": {"arg": 123, },
        # task definition overrides, currently not supported
        "task_override": None,
    },
    "step2": {
        # identify the node, so that we code reference outputs, use only alphanumeric characters
        "node_name": "step2",
        # parent node, to be executed before this step
        "parent_node": "step1",
        # the experiment/task id to clone & execute
        "base_task_id": "123456aa",
        # preferred queue name/id to use for execution
        "queue": "2xgpu",
        # preferred docker image (override experiment request)
        "docker": None,
        # parameter overrides
        "parameter_override": {
            # plug the output of pipeline node `step1` artifact named `my_data` into the Task parameter `url`
            "url": "@step1:artifacts/my_data",
            # plug the output of pipeline node `step1` parameter named `arg` into the Task parameter `arg`
            "arg": "@step1:parameters/arg",
        },
        # task definition overrides, currently not supported
        "task_override": None,
    },
    "step3": {
        # identify the node, so that we code reference outputs, use only alphanumeric characters
        "node_name": "step3",
        # parent node, to be executed before this step
        "parent_node": "step2",
        # the experiment/task id to clone & execute
        "base_task_id": "zzcc1244",
        # preferred queue name/id to use for execution
        "queue": "2xGPUS",
        # preferred docker image (override experiment request)
        "docker": None,
        # parameter overrides
        "parameter_override": {
            # plug the output of pipeline node `step2` last output model into the Task parameter url
            "model_url": "@step2:models/output/-1",
        },
        # task definition overrides, currently not supported
        "task_override": None,
    },
}
