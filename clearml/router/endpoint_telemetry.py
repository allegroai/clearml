import copy
import time
import uuid
from threading import Thread

from ..task import Task
from ..utilities.resource_monitor import ResourceMonitor


class EndpointTelemetry:
    BACKEND_STAT_MAP = {
        "cpu_usage_*": "cpu_usage",
        "cpu_temperature_*": "cpu_temperature",
        "disk_free_percent": "disk_free_home",
        "io_read_mbs": "disk_read",
        "io_write_mbs": "disk_write",
        "network_tx_mbs": "network_tx",
        "network_rx_mbs": "network_rx",
        "memory_free_gb": "memory_free",
        "memory_used_gb": "memory_used",
        "gpu_temperature_*": "gpu_temperature",
        "gpu_mem_used_gb_*": "gpu_memory_used",
        "gpu_mem_free_gb_*": "gpu_memory_free",
        "gpu_utilization_*": "gpu_usage",
    }

    def __init__(
        self,
        endpoint_name="endpoint",
        model_name="model",
        model=None,
        model_url=None,
        model_source=None,
        model_version=None,
        app_id=None,
        app_instance=None,
        tags=None,
        system_tags=None,
        container_id=None,
        input_size=None,
        input_type="str",
        report_statistics=True,
        endpoint_url=None,
        preprocess_artifact=None,
        force_register=False,
    ):
        self.report_window = 30
        self._previous_readouts = {}
        self._previous_readouts_ts = time.time()
        self._num_readouts = 0
        self.container_info = {
            "container_id": container_id or str(uuid.uuid4()).replace("-", ""),
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "model_source": model_source,
            "model_version": model_version,
            "preprocess_artifact": preprocess_artifact,
            "input_type": str(input_type),
            "input_size": str(input_size),
            "tags": tags,
            "system_tags": system_tags,
            "endpoint_url": endpoint_url,
        }
        references = []
        if app_id:
            references.append({"type": "app_id", "value": app_id})
            if app_instance:
                references.append({"type": "app_instance", "value": app_instance})
        references.append({"type": "task", "value": Task.current_task().id})
        if model:
            references.append({"type": "model", "value": model})
        if model_url:
            references.append({"type": "url", "value": model_url})
        self.container_info["reference"] = references
        self.session = Task._get_default_session()
        self.requests_num = 0
        self.requests_num_window = 0
        self.requests_num_prev_window = 0
        self.latency_sum_window = 0
        self.uptime_timestamp = time.time()
        self.last_request_time = None
        # use readily available resource monitor, otherwise create one (can happen in spawned subprocesses)
        self.resource_monitor = Task.current_task()._resource_monitor or ResourceMonitor(Task.current_task())
        if not container_id and not force_register:
            self.register_container()
        self._stop_container_status_report_daemon = False
        if report_statistics:
            Thread(target=self.container_status_report_daemon, daemon=True).start()

    def stop(self):
        self._stop_container_status_report_daemon = True

    def update(
        self,
        endpoint_name=None,
        model_name=None,
        model=None,
        model_url=None,
        model_source=None,
        model_version=None,
        tags=None,
        system_tags=None,
        input_size=None,
        input_type=None,
        endpoint_url=None,
        preprocess_artifact=None,
    ):
        update_dict = {}
        if endpoint_name is not None:
            update_dict["endpoint_name"] = endpoint_name
        if model_name is not None:
            update_dict["model_name"] = model_name
        if model_source is not None:
            update_dict["model_source"] = model_source
        if model_version is not None:
            update_dict["model_version"] = model_version
        if preprocess_artifact is not None:
            update_dict["preprocess_artifact"] = preprocess_artifact
        if input_type is not None:
            update_dict["input_type"] = input_type
        if input_size is not None:
            update_dict["input_size"] = input_size
        if tags is not None:
            update_dict["tags"] = tags
        if system_tags is not None:
            update_dict["system_tags"] = system_tags
        if endpoint_url is not None:
            update_dict["endpoint_url"] = endpoint_url
        self.container_info.update(update_dict)
        references_to_add = {}
        if model:
            references_to_add["model"] = {"type": "model", "value": model}
        if model_url:
            references_to_add["model_url"] = {"type": "url", "value": model_url}
        for reference in self.container_info["reference"]:
            if reference["type"] in references_to_add:
                reference["value"] = references_to_add[reference["type"]]["value"]
                references_to_add.pop(reference["type"], None)
        self.container_info["reference"].extend(list(references_to_add.values()))

    def register_container(self):
        result = self.session.send_request("serving", "register_container", json=self.container_info)
        if result.status_code != 200:
            print("Failed registering container: {}".format(result.json()))

    def wait_for_endpoint_url(self):
        while not self.container_info.get("endpoint_url"):
            Task.current_task().reload()
            endpoint = Task.current_task()._get_runtime_properties().get("endpoint")
            if endpoint:
                self.container_info["endpoint_url"] = endpoint
                self.uptime_timestamp = time.time()
            else:
                time.sleep(1)

    def get_machine_stats(self):
        def create_general_key(old_key):
            return "{}_*".format(old_key)

        stats = self.resource_monitor._machine_stats()
        elapsed = time.time() - self._previous_readouts_ts
        self._previous_readouts_ts = time.time()
        updates = {}
        for k, v in stats.items():
            if k.endswith("_mbs"):
                v = (v - self._previous_readouts.get(k, v)) / elapsed
                updates[k] = v
        self._previous_readouts = copy.deepcopy(stats)
        stats.update(updates)
        self._num_readouts += 1

        preprocessed_stats = {}
        ordered_keys = sorted(stats.keys())
        for k in ordered_keys:
            v = stats[k]
            if k in ["memory_used_gb", "memory_free_gb"]:
                v *= 1024
            if isinstance(v, float):
                v = round(v, 3)
            stat_key = self.BACKEND_STAT_MAP.get(k)
            if stat_key:
                preprocessed_stats[stat_key] = v
            else:
                general_key = create_general_key(k)
                if general_key.startswith("gpu"):
                    prev_general_key = general_key
                    general_key = "_".join(["gpu"] + general_key.split("_")[2:])
                    if general_key == "gpu_mem_used_gb_*":
                        gpu_index = prev_general_key.split("_")[1]
                        mem_usage = min(stats["gpu_{}_mem_usage".format(gpu_index)], 99.99)
                        mem_free = stats["gpu_{}_mem_free_gb".format(gpu_index)]
                        v = (mem_usage * mem_free) / (100 - mem_usage)
                    if general_key in ["gpu_mem_used_gb_*", "gpu_mem_free_gb_*"]:
                        v *= 1024
                general_key = self.BACKEND_STAT_MAP.get(general_key)
                if general_key:
                    preprocessed_stats.setdefault(general_key, []).append(v)
        return preprocessed_stats

    def container_status_report_daemon(self):
        while not self._stop_container_status_report_daemon:
            self.container_status_report()
            time.sleep(self.report_window)

    def container_status_report(self):
        self.wait_for_endpoint_url()
        status_report = {**self.container_info}
        status_report["uptime_sec"] = int(time.time() - self.uptime_timestamp)
        status_report["requests_num"] = self.requests_num
        status_report["requests_min"] = self.requests_num_window + self.requests_num_prev_window
        status_report["latency_ms"] = (
            0 if (self.requests_num_window == 0) else (self.latency_sum_window / self.requests_num_window)
        )
        status_report["machine_stats"] = self.get_machine_stats()
        self.requests_num_prev_window = self.requests_num_window
        self.requests_num_window = 0
        self.latency_sum_window = 0
        self.latency_num_window = 0
        result = self.session.send_request("serving", "container_status_report", json=status_report)
        if result.status_code != 200:
            print("Failed sending status report: {}".format(result.json()))

    def update_last_request_time(self):
        self.last_request_time = time.time()

    def update_statistics(self):
        self.requests_num += 1
        self.requests_num_window += 1
        latency = (time.time() - self.last_request_time) * 1000
        self.latency_sum_window += latency

    def on_request(self):
        self.update_last_request_time()

    def on_response(self):
        self.update_statistics()
