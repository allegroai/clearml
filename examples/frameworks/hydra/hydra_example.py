# ClearML - Hydra Example
#
import hydra

from omegaconf import OmegaConf

from clearml import Task


@hydra.main(config_path="config_files", config_name="config", version_base=None)
def my_app(cfg):
    # type (DictConfig) -> None
    task = Task.init(project_name="examples", task_name="Hydra configuration")
    logger = task.get_logger()
    logger.report_text("You can view your full hydra configuration under Configuration tab in the UI")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
