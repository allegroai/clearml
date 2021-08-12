import base64
from typing import Union

import attr

from .auto_scaler import AutoScaler
from .. import Task
from ..utilities.pyhocon import ConfigTree, ConfigFactory

try:
    # noinspection PyPackageRequirements
    import boto3

    Task.add_requirements("boto3")
except ImportError:
    raise ValueError(
        "AwsAutoScaler requires 'boto3' package, it was not found\n"
        "install with: pip install boto3"
    )


class AwsAutoScaler(AutoScaler):
    @attr.s
    class Settings(AutoScaler.Settings):
        workers_prefix = attr.ib(default="dynamic_aws")
        cloud_provider = attr.ib(default="AWS")

    startup_bash_script = [
        "#!/bin/bash",
        "sudo apt-get update",
        "sudo apt-get install -y python3-dev",
        "sudo apt-get install -y python3-pip",
        "sudo apt-get install -y gcc",
        "sudo apt-get install -y git",
        "sudo apt-get install -y build-essential",
        "python3 -m pip install -U pip",
        "python3 -m pip install virtualenv",
        "python3 -m virtualenv clearml_agent_venv",
        "source clearml_agent_venv/bin/activate",
        "python -m pip install clearml-agent",
        "echo 'agent.git_user=\"{git_user}\"' >> /root/clearml.conf",
        "echo 'agent.git_pass=\"{git_pass}\"' >> /root/clearml.conf",
        "echo \"{clearml_conf}\" >> /root/clearml.conf",
        "export CLEARML_API_HOST={api_server}",
        "export CLEARML_WEB_HOST={web_server}",
        "export CLEARML_FILES_HOST={files_server}",
        "export DYNAMIC_INSTANCE_ID=`curl http://169.254.169.254/latest/meta-data/instance-id`",
        "export CLEARML_WORKER_ID={worker_id}:$DYNAMIC_INSTANCE_ID",
        "export CLEARML_API_ACCESS_KEY='{access_key}'",
        "export CLEARML_API_SECRET_KEY='{secret_key}'",
        "source ~/.bashrc",
        "{bash_script}",
        "python -m clearml_agent --config-file '/root/clearml.conf' daemon --queue '{queue}' {docker}",
        "shutdown",
    ]

    def __init__(self, settings, configuration):
        # type: (Union[dict, AwsAutoScaler.Settings], Union[dict, AwsAutoScaler.Configuration]) -> None
        super(AwsAutoScaler, self).__init__(settings, configuration)

    def spin_up_worker(self, resource, worker_id_prefix, queue_name):
        """
        Creates a new worker for clearml.
        First, create an instance in the cloud and install some required packages.
        Then, define clearml-agent environment variables and run clearml-agent for the specified queue.
        NOTE: - Will wait until instance is running
              - This implementation assumes the instance image already has docker installed

        :param str resource: resource name, as defined in BUDGET and QUEUES.
        :param str worker_id_prefix: worker name prefix
        :param str queue_name: clearml queue to listen to
        """
        resource_conf = self.resource_configurations[resource]
        # Add worker type and AWS instance type to the worker name.
        worker_id = "{worker_id_prefix}:{worker_type}:{instance_type}".format(
            worker_id_prefix=worker_id_prefix,
            worker_type=resource,
            instance_type=resource_conf["instance_type"],
        )

        # user_data script will automatically run when the instance is started. it will install the required packages
        # for clearml-agent configure it using environment variables and run clearml-agent on the required queue
        user_data = ('\n'.join(self.startup_bash_script) + '\n').format(
            api_server=self.api_server,
            web_server=self.web_server,
            files_server=self.files_server,
            worker_id=worker_id,
            access_key=self.access_key,
            secret_key=self.secret_key,
            queue=queue_name,
            git_user=self.git_user or "",
            git_pass=self.git_pass or "",
            clearml_conf='\\"'.join(self.extra_clearml_conf.split('"')),
            bash_script=self.extra_vm_bash_script,
            docker="--docker '{}'".format(self.default_docker_image) if self.default_docker_image else "",
        )

        ec2 = boto3.client(
            "ec2",
            aws_access_key_id=self.cloud_credentials_key or None,
            aws_secret_access_key=self.cloud_credentials_secret or None,
            region_name=self.cloud_credentials_region,
        )

        launch_specification = ConfigFactory.from_dict(
            {
                "ImageId": resource_conf["ami_id"],
                "InstanceType": resource_conf["instance_type"],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": resource_conf["ebs_device_name"],
                        "Ebs": {
                            "VolumeSize": resource_conf["ebs_volume_size"],
                            "VolumeType": resource_conf["ebs_volume_type"],
                        },
                    }
                ],
                "Placement": {"AvailabilityZone": resource_conf["availability_zone"]},
            }
        )
        if resource_conf.get("key_name", None):
            launch_specification["KeyName"] = resource_conf["key_name"]
        if resource_conf.get("security_group_ids", None):
            launch_specification["SecurityGroupIds"] = resource_conf[
                "security_group_ids"
            ]

        if resource_conf["is_spot"]:
            # Create a request for a spot instance in AWS
            encoded_user_data = base64.b64encode(user_data.encode("ascii")).decode(
                "ascii"
            )
            launch_specification["UserData"] = encoded_user_data
            ConfigTree.merge_configs(
                launch_specification, resource_conf.get("extra_configurations", {})
            )

            instances = ec2.request_spot_instances(
                LaunchSpecification=launch_specification
            )

            # Wait until spot request is fulfilled
            request_id = instances["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
            waiter = ec2.get_waiter("spot_instance_request_fulfilled")
            waiter.wait(SpotInstanceRequestIds=[request_id])
            # Get the instance object for later use
            response = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )
            instance_id = response["SpotInstanceRequests"][0]["InstanceId"]

        else:
            # Create a new EC2 instance
            launch_specification.update(
                MinCount=1,
                MaxCount=1,
                UserData=user_data,
                InstanceInitiatedShutdownBehavior="terminate",
            )
            ConfigTree.merge_configs(
                launch_specification, resource_conf.get("extra_configurations", {})
            )

            instances = ec2.run_instances(**launch_specification)

            # Get the instance object for later use
            instance_id = instances["Instances"][0]["InstanceId"]

        instance = boto3.resource(
            "ec2",
            aws_access_key_id=self.cloud_credentials_key or None,
            aws_secret_access_key=self.cloud_credentials_secret or None,
            region_name=self.cloud_credentials_region,
        ).Instance(instance_id)

        # Wait until instance is in running state
        instance.wait_until_running()

    # Cloud-specific implementation (currently, only AWS EC2 is supported)
    def spin_down_worker(self, instance_id):
        """
        Destroys the cloud instance.

        :param instance_id: Cloud instance ID to be destroyed (currently, only AWS EC2 is supported)
        :type instance_id: str
        """
        try:
            boto3.resource(
                "ec2",
                aws_access_key_id=self.cloud_credentials_key or None,
                aws_secret_access_key=self.cloud_credentials_secret or None,
                region_name=self.cloud_credentials_region,
            ).instances.filter(InstanceIds=[instance_id]).terminate()
        except Exception as ex:
            raise ex
