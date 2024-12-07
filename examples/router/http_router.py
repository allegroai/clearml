"""
Example on how to use the ClearML HTTP router.
For this example, you would first need a webserver to route the traffic to:
`simple_webserver.py` launches such a server. Running the script will start a
webserver, bound to localhost:8000.

Then, when running this example, it creates a router which binds to 0.0.0.0:9000.
A local route is then created, which will proxy all traffic from
`http://<PRIVATE_IP>:9000/example_source` to `http://localhost:8000/serve`.

Traffic can be intercepted both on request and response via callbacks. See
`request_callback` and `response_callback`.

By default, the route traffic is monitored and telemetry is sent to the ClearML
server. To disable this, pass `endpoint_telemetry=False` when creating the route
"""

import time
from clearml import Task


def request_callback(request, persistent_state):
    persistent_state["last_request_time"] = time.time()


def response_callback(response, request, persistent_state):
    print("Latency:", time.time() - persistent_state["last_request_time"])


if __name__ == "__main__":
    task = Task.init(project_name="Router Example", task_name="Router Example")
    router = task.get_http_router()
    router.set_local_proxy_parameters(incoming_port=9000, default_target="http://localhost:8000")
    router.create_local_route(
        source="/example_source",
        target="http://localhost:8000/serve",  # route traffic to this address
        request_callback=request_callback,  # intercept requests
        response_callback=response_callback,  # intercept responses
        endpoint_telemetry={"model": "MyModel"}  # set this to False to disable telemetry
    )
    router.deploy(wait=True)
    # run `curl http://localhost:9000/example_source/1`
