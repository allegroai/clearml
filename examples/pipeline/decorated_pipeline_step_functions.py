from clearml import PipelineController


def our_decorator(func):
    def function_wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1
    return function_wrapper


@our_decorator
def step():
    return 1


def evaluate(step_return):
    assert step_return == 2


if __name__ == "__main__":
    pipeline = PipelineController(name="test_decorated", project="test_decorated")
    pipeline.add_function_step(name="step", function=step, function_return=["step_return"])
    pipeline.add_function_step(
        name="evaluate",
        function=evaluate,
        function_kwargs=dict(step_return='${step.step_return}')
    )
    pipeline.start_locally(run_pipeline_steps_locally=True)
