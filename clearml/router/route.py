import inspect
from .endpoint_telemetry import EndpointTelemetry


class Route:
    def __init__(self, target_url, request_callback=None, response_callback=None, session=None, error_callback=None):
        self.target_url = target_url
        self.request_callback = request_callback
        self.response_callback = response_callback
        self.error_callback = error_callback
        self.session = session
        self.persistent_state = {}
        self._endpoint_telemetry = None
        self._endpoint_telemetry_args = None

    def set_endpoint_telemetry_args(
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
        force_register=False
    ):
        self._endpoint_telemetry_args = dict(
            endpoint_name=endpoint_name,
            model_name=model_name,
            model=model,
            model_url=model_url,
            model_source=model_source,
            model_version=model_version,
            app_id=app_id,
            app_instance=app_instance,
            tags=tags,
            system_tags=system_tags,
            container_id=container_id,
            input_size=input_size,
            input_type=input_type,
            report_statistics=report_statistics,
            endpoint_url=endpoint_url,
            preprocess_artifact=preprocess_artifact,
            force_register=force_register
        )

    def start_endpoint_telemetry(self):
        if self._endpoint_telemetry is not None or self._endpoint_telemetry_args is None:
            return
        self._endpoint_telemetry = EndpointTelemetry(**self._endpoint_telemetry_args)

    def stop_endpoint_telemetry(self):
        if self._endpoint_telemetry is None:
            return
        self._endpoint_telemetry.stop()
        self._endpoint_telemetry = None

    async def on_request(self, request):
        new_request = request
        if self.request_callback:
            new_request = self.request_callback(request, persistent_state=self.persistent_state) or request
            if inspect.isawaitable(new_request):
                new_request = (await new_request) or request
        if self._endpoint_telemetry:
            self._endpoint_telemetry.on_request()
        return new_request

    async def on_response(self, response, request):
        new_response = response
        if self.response_callback:
            new_response = self.response_callback(response, request, persistent_state=self.persistent_state) or response
            if inspect.isawaitable(new_response):
                new_response = (await new_response) or response
        if self._endpoint_telemetry:
            self._endpoint_telemetry.on_response()
        return new_response

    async def on_error(self, request, error):
        on_error_result = None
        if self.error_callback:
            on_error_result = self.error_callback(request, error, persistent_state=self.persistent_state)
            if inspect.isawaitable(on_error_result):
                await on_error_result
        return on_error_result
