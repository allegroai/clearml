from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import yaml
from six.moves import input

from trains import Task
from trains.automation.aws_auto_scaler import AwsAutoScaler
from trains.config import running_remotely
from trains.utilities.wizard.user_input import get_input, input_int, input_bool

CONF_FILE = "aws_autoscaler.yaml"
DEFAULT_DOCKER_IMAGE = "nvidia/cuda"


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--run",
        help="Run the autoscaler after wizard finished",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if running_remotely():
        hyper_params = AwsAutoScaler.Settings().as_dict()
        configurations = AwsAutoScaler.Configuration().as_dict()
    else:
        print("AWS Autoscaler setup\n")

        config_file = Path(CONF_FILE).absolute()
        if config_file.exists() and input_bool(
            "Load configurations from config file '{}' [Y/n]? ".format(str(CONF_FILE)),
            default=True,
        ):
            with config_file.open("r") as f:
                conf = yaml.load(f, Loader=yaml.SafeLoader)
            hyper_params = conf["hyper_params"]
            configurations = conf["configurations"]
        else:
            configurations, hyper_params = run_wizard()

            try:
                with config_file.open("w+") as f:
                    conf = {
                        "hyper_params": hyper_params,
                        "configurations": configurations,
                    }
                    yaml.safe_dump(conf, f)
            except Exception:
                print(
                    "Error! Could not write configuration file at: {}".format(
                        str(CONF_FILE)
                    )
                )
                return

    task = Task.init(project_name="Auto-Scaler", task_name="AWS Auto-Scaler")
    task.connect(hyper_params)
    task.connect_configuration(configurations)

    autoscaler = AwsAutoScaler(hyper_params, configurations)

    if running_remotely() or args.run:
        autoscaler.start()


def run_wizard():
    # type: () -> Tuple[dict, dict]

    hyper_params = AwsAutoScaler.Settings()
    configurations = AwsAutoScaler.Configuration()

    hyper_params.cloud_credentials_key = get_input("AWS Access Key ID", required=True)
    hyper_params.cloud_credentials_secret = get_input(
        "AWS Secret Access Key", required=True
    )
    hyper_params.cloud_credentials_region = get_input("AWS region name", required=True)
    # get GIT User/Pass for cloning
    print(
        "\nGIT credentials:"
        "\nEnter GIT username for repository cloning (leave blank for SSH key authentication): [] ",
        end="",
    )
    git_user = input()
    if git_user.strip():
        print("Enter password for user '{}': ".format(git_user), end="")
        git_pass = input()
        print(
            "Git repository cloning will be using user={} password={}".format(
                git_user, git_pass
            )
        )
    else:
        git_user = None
        git_pass = None

    hyper_params.git_user = git_user
    hyper_params.git_pass = git_pass

    hyper_params.default_docker_image = get_input(
        "default docker image/parameters",
        "to use [default is {}]".format(DEFAULT_DOCKER_IMAGE),
        default=DEFAULT_DOCKER_IMAGE,
        new_line=True,
    )
    print("\nDefine the type of machines you want the autoscaler to use")
    resource_configurations = {}
    while True:
        resource_name = get_input(
            "machine type name",
            "(remember it, we will later use it in the budget section)",
            required=True,
            new_line=True,
        )
        resource_configurations[resource_name] = {
            "instance_type": get_input(
                "instance type",
                "for resource '{}' [default is 'g4dn.4xlarge']".format(resource_name),
                default="g4dn.4xlarge",
            ),
            "is_spot": input_bool(
                "is '{}' resource using spot instances? [t/F]".format(resource_name)
            ),
            "availability_zone": get_input(
                "availability zone",
                "for resource '{}' [default is 'us-east-1b']".format(resource_name),
                default="us-east-1b",
            ),
            "ami_id": get_input(
                "ami_id",
                "for resource '{}' [default is 'ami-07c95cafbb788face']".format(
                    resource_name
                ),
                default="ami-07c95cafbb788face",
            ),
            "ebs_device_name": get_input(
                "ebs_device_name",
                "for resource '{}' [default is '/dev/xvda']".format(resource_name),
                default="/dev/xvda",
            ),
            "ebs_volume_size": input_int(
                "ebs_volume_size",
                " for resource '{}' [default is '100']".format(resource_name),
                default=100,
            ),
            "ebs_volume_type": get_input(
                "ebs_volume_type",
                "for resource '{}' [default is 'gp2']".format(resource_name),
                default="gp2",
            ),
        }
        if not input_bool("\nDefine another resource? [y/N]"):
            break
    configurations.resource_configurations = resource_configurations

    configurations.extra_vm_bash_script = input(
        "\nEnter any pre-execution bash script to be executed on the newly created instances: "
    )

    print("\nSet up the budget\n")
    queues = defaultdict(list)
    while True:
        queue_name = get_input("queue name", required=True)
        while True:
            queue_type = get_input(
                "queue type",
                "(use the resources names defined earlier)",
                required=True,
            )
            max_instances = input_int(
                "maximum number of instances allowed", required=True
            )
            queues[queue_name].append((queue_type, max_instances))

            if not input_bool("\nAdd another type to queue? [y/N]: "):
                break
        if not input_bool("Define another queue? [y/N]: "):
            break
    configurations.queues = dict(queues)

    hyper_params.max_idle_time_min = input_int(
        "maximum idle time",
        "for the autoscaler (in minutes, default is 15)",
        default=15,
        new_line=True,
    )
    hyper_params.polling_interval_time_min = input_int(
        "polling interval", "for the autoscaler (in minutes, default is 5)", default=5,
    )

    return configurations.as_dict(), hyper_params.as_dict()


if __name__ == "__main__":
    main()
