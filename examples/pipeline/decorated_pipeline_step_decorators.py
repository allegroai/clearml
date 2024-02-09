from clearml import PipelineDecorator


def our_decorator(func):
    def function_wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1
    return function_wrapper


@PipelineDecorator.component()
@our_decorator
def step():
    return 1


@PipelineDecorator.pipeline(name="test_decorated", project="test_decorated")
def pipeline():
    result = step()
    assert result == 2


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline()
