from .fastapi_proxy import FastAPIProxy


class HttpProxy:
    DEFAULT_PORT = 9000

    def __init__(self, port=None, workers=None, default_target=None, log_level=None, access_log=True):
        # at the moment, only a fastapi proxy is supported
        self.base_proxy = FastAPIProxy(
            port or self.DEFAULT_PORT,
            workers=workers,
            default_target=default_target,
            log_level=log_level,
            access_log=access_log,
        )
        self.base_proxy.start()
        self.port = port
        self.routes = {}

    def add_route(
        self,
        source,
        target,
        request_callback=None,
        response_callback=None,
        endpoint_telemetry=True,
        error_callback=None,
    ):
        self.routes[source] = self.base_proxy.add_route(
            source=source,
            target=target,
            request_callback=request_callback,
            response_callback=response_callback,
            endpoint_telemetry=endpoint_telemetry,
            error_callback=error_callback,
        )
        return self.routes[source]

    def remove_route(self, source):
        self.routes.pop(source, None)
        self.base_proxy.remove_route(source)

    def get_routes(self):
        return self.routes

    def start(self):
        self.base_proxy.start()

    def stop(self):
        self.base_proxy.stop()
