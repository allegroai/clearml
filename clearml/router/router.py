from typing import Optional, Callable, Dict, Union  # noqa
from fastapi import Request, Response  # noqa
from .proxy import HttpProxy


class HttpRouter:
    """
    A router class to manage HTTP routing for an application.
    Allows the creation, deployment, and management of local and external endpoints,
    as well as the configuration of a local proxy for traffic routing.

    Example usage:

    .. code-block:: py
        def request_callback(request, persistent_state):
            persistent_state["last_request_time"] = time.time()

        def response_callback(response, request, persistent_state):
            print("Latency:", time.time() - persistent_state["last_request_time"])
            if urllib.parse.urlparse(str(request.url).rstrip("/")).path == "/modify":
                new_content = response.body.replace(b"modify", b"modified")
                headers = copy.deepcopy(response.headers)
                headers["Content-Length"] = str(len(new_content))
                return Response(status_code=response.status_code, headers=headers, content=new_content)

        router = Task.current_task().get_http_router()
        router.set_local_proxy_parameters(incoming_port=9000)
        router.create_local_route(
            source="/",
            target="http://localhost:8000",
            request_callback=request_callback,
            response_callback=response_callback,
            endpoint_telemetry={"model": "MyModel"}
        )
        router.deploy(wait=True)
    """
    _instance = None

    def __init__(self, task):
        """
        Do not use directly. Use `Task.get_router` instead
        """
        self._task = task
        self._external_endpoint_port = None
        self._proxy = None
        self._proxy_params = {"port": HttpProxy.DEFAULT_PORT}

    def set_local_proxy_parameters(self, incoming_port=None, default_target=None):
        # type: (Optional[int], Optional[str]) -> ()
        """
        Set the parameters with which the local proxy is initialized

        :param incoming_port: The incoming port of the proxy
        :param default_target: If None, no default target is set. Otherwise, route all traffic
            that doesn't match a local route created via `create_local_route` to this target
        """
        self._proxy_params["port"] = incoming_port or HttpProxy.DEFAULT_PORT
        self._proxy_params["default_target"] = default_target

    def start_local_proxy(self):
        """
        Start the local proxy without deploying the router, i.e. requesting an external endpoint
        """
        self._proxy = self._proxy or HttpProxy(**self._proxy_params)

    def create_local_route(
        self,
        source,  # type: str
        target,  # type: str
        request_callback=None,  # type: Callable[Request, Dict]
        response_callback=None,  # type: Callable[Response, Request, Dict]
        endpoint_telemetry=True  # type: Union[bool, Dict]
    ):
        """
        Create a local route from a source to a target through a proxy. If no proxy instance
        exists, one is automatically created.
        This function enables routing traffic between the source and target endpoints, optionally
        allowing custom callbacks to handle requests and responses or to gather telemetry data
        at the endpoint.
        To customize proxy parameters, use the `Router.set_local_proxy_parameters` method.
        By default, the proxy binds to port 9000 for incoming requests.

        :param source: The source path for routing the traffic. For example, `/` will intercept
            all the traffic sent to the proxy, while `/example` will only intercept the calls
            that have `/example` as the path prefix.
        :param target: The target URL where the intercepted traffic is routed.
        :param request_callback: A function used to process each request before it is forwarded to the target.
            The callback must have the following parameters:
            - request - The intercepted FastAPI request
            - persistent_state - A dictionary meant to be used as a caching utility object.
            Shared with `response_callback`
            The callback can return a FastAPI Request, in which case this request will be forwarded to the target
        :param response_callback: A function used to process each response before it is returned by the proxy.
            The callback must have the following parameters:
            - response - The FastAPI response
            - request - The FastAPI request (after being preprocessed by the proxy)
            - persistent_state - A dictionary meant to be used as a caching utility object.
            Shared with `request_callback`
            The callback can return a FastAPI Response, in which case this response will be forwarded
        :param endpoint_telemetry: If True, enable endpoint telemetry. If False, disable it.
            If a dictionary is passed, enable endpoint telemetry with custom parameters.
            The parameters are:
            - endpoint_url - URL to the endpoint, mandatory if no external URL has been requested
            - endpoint_name - name of the endpoint
            - model_name - name of the model served by the endpoint
            - model - referenced model
            - model_url - URL to the model
            - model_source - Source of the model
            - model_version - Model version
            - app_id - App ID, if used inside a ClearML app
            - app_instance - The ID of the instance the ClearML app is running
            - tags - ClearML tags
            - system_tags - ClearML system tags
            - container_id - Container ID, should be unique
            - input_size - input size of the model
            - input_type - input type expected by the model/endpoint
            - report_statistics - whether or not to report statistics
        """
        self.start_local_proxy()
        self._proxy.add_route(
            source,
            target,
            request_callback=request_callback,
            response_callback=response_callback,
            endpoint_telemetry=endpoint_telemetry,
        )

    def remove_local_route(self, source):
        # type: (str) -> ()
        """
        Remove a local route. If endpoint telemetry is enabled for that route, disable it

        :param source: Remove route based on the source path used to route the traffic
        """
        if self._proxy:
            self._proxy.remove_route(source)

    def deploy(
        self, wait=False, wait_interval_seconds=3.0, wait_timeout_seconds=90.0
    ):
        # type: (Optional[int], str, bool, float, float) -> Optional[Dict]
        """
        Start the local HTTP proxy and request an external endpoint for an application

        :param port: Port the application is listening to. If no port is supplied, a local proxy
            will be created. To control the proxy parameters, use `Router.set_local_proxy_parameters`.
            To control create local routes through the proxy, use `Router.create_local_route`.
            By default, the incoming port bound by the proxy is 9000
        :param protocol: As of now, only `http` is supported
        :param wait: If True, wait for the endpoint to be assigned
        :param wait_interval_seconds: The poll frequency when waiting for the endpoint
        :param wait_timeout_seconds: If this timeout is exceeded while waiting for the endpoint,
            the method will no longer wait and None will be returned

        :return: If wait is False, this method will return None.
            If no endpoint could be found while waiting, this mehtod returns None.
            Otherwise, it returns a dictionary containing the following values:
            - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
            - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
            - port - the port exposed by the application
            - protocol - the protocol used by the endpo"int
        """
        self._proxy = self._proxy or HttpProxy(**self._proxy_params)
        return self._task.request_external_endpoint(
            port=self._proxy.port,
            protocol="http",
            wait=wait,
            wait_interval_seconds=wait_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def wait_for_external_endpoint(self, wait_interval_seconds=3.0, wait_timeout_seconds=90.0):
        # type: (float) -> Optional[Dict]
        """
        Wait for an external endpoint to be assigned

        :param wait_interval_seconds: The poll frequency when waiting for the endpoint
        :param wait_timeout_seconds: If this timeout is exceeded while waiting for the endpoint,
            the method will no longer wait

        :return: If no endpoint could be found while waiting, this mehtod returns None.
            Otherwise, it returns a dictionary containing the following values:
            - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
            - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
            - port - the port exposed by the application
            - protocol - the protocol used by the endpoint
        """
        return self._task.wait_for_external_endpoint(
            protocol="http", wait_interval_seconds=wait_interval_seconds, wait_timeout_seconds=wait_timeout_seconds
        )

    def list_external_endpoints(self):
        # type: () -> List[Dict]
        """
        List all external endpoints assigned

        :return: A list of dictionaries. Each dictionary contains the following values:
            - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
            - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
            - port - the port exposed by the application
            - protocol - the protocol used by the endpoint
        """
        return self._task.list_external_endpoints(protocol="http")
