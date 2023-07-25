try:
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
except ImportError:
    import sys
    print("Module 'lightning' not installed (only available for Python 3.8+)")
    sys.exit(0)
from clearml import Task


if __name__ == "__main__":
    Task.add_requirements("requirements.txt")
    Task.init(project_name="example", task_name="pytorch_lightning_jsonargparse")
    LightningCLI(DemoModel, BoringDataModule)
