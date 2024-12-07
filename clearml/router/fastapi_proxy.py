from fastapi import FastAPI, Request, Response
from typing import Optional
from multiprocessing import Process
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match
import functools
import threading
import httpx
import uvicorn
from .route import Route
from ..utilities.process.mp import SafeQueue


class FastAPIProxy:
    ALL_REST_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

    def __init__(self, port, workers=None, default_target=None):
        self.app = None
        self.routes = {}
        self.port = port
        self.message_queue = SafeQueue()
        self.uvicorn_subprocess = None
        self.workers = workers
        self._default_target = default_target
        self._default_session = None
        self._in_subprocess = False

    def _create_default_route(self):
        proxy = self

        class DefaultRouteMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                scope = {
                    "type": "http",
                    "method": request.method,
                    "path": request.url.path,
                    "root_path": "",
                    "headers": request.headers.raw,
                    "query_string": request.url.query.encode("utf-8"),
                    "client": request.client,
                    "server": request.scope.get("server"),
                    "scheme": request.url.scheme,
                    "extensions": request.scope.get("extensions", {}),
                    "app": request.scope.get("app"),
                }
                for route in proxy.app.router.routes:
                    if route.matches(scope)[0] == Match.FULL:
                        return await call_next(request)
                proxied_response = await proxy._default_session.request(
                    method=request.method,
                    url=proxy._default_target + request.url.path,
                    headers=dict(request.headers),
                    content=await request.body(),
                    params=request.query_params,
                )
                return Response(
                    content=proxied_response.content,
                    headers=dict(proxied_response.headers),
                    status_code=proxied_response.status_code,
                )
        self.app.add_middleware(DefaultRouteMiddleware)

    async def proxy(
        self,
        request: Request,
        path: Optional[str] = None,
        source_path: Optional[str] = None,
    ):
        route_data = self.routes.get(source_path)
        if not route_data:
            return Response(status_code=404)

        request = route_data.on_request(request)
        proxied_response = await route_data.session.request(
            method=request.method,
            url=f"{route_data.target_url}/{path}" if path else route_data.target_url,
            headers=dict(request.headers),
            content=await request.body(),
            params=request.query_params,
        )
        proxied_response = Response(
            content=proxied_response.content,
            headers=dict(proxied_response.headers),
            status_code=proxied_response.status_code,
        )
        return route_data.on_response(proxied_response, request)

    def add_route(
        self,
        source,
        target,
        request_callback=None,
        response_callback=None,
        endpoint_telemetry=True
    ):
        if not self._in_subprocess:
            self.message_queue.put(
                {
                    "method": "add_route",
                    "kwargs": {
                        "source": source,
                        "target": target,
                        "request_callback": request_callback,
                        "response_callback": response_callback,
                        "endpoint_telemetry": endpoint_telemetry
                    },
                }
            )
            return
        should_add_route = False
        if source not in self.routes:
            should_add_route = True
        else:
            self.routes[source].stop_endpoint_telemetry()
        self.routes[source] = Route(
            target,
            request_callback=request_callback,
            response_callback=response_callback,
            session=httpx.AsyncClient(timeout=None),
        )
        if endpoint_telemetry is True:
            endpoint_telemetry = {}
        if endpoint_telemetry is not False:
            self.routes[source].set_endpoint_telemetry_args(**endpoint_telemetry)
        if self._in_subprocess:
            self.routes[source].start_endpoint_telemetry()
        if should_add_route:
            self.app.add_api_route(
                source,
                functools.partial(
                    self.proxy,
                    source_path=source,
                ),
                methods=self.ALL_REST_METHODS,
            )
            self.app.add_api_route(
                source.rstrip("/") + "/{path:path}",
                functools.partial(
                    self.proxy,
                    source_path=source,
                ),
                methods=self.ALL_REST_METHODS,
            )
        return self.routes[source]

    def remove_route(self, source):
        if not self._in_subprocess:
            self.message_queue.put({"method": "remove_route", "kwargs": {"source": source}})
            return
        route = self.routes.get(source)
        if route:
            route.stop_endpoint_telemetry()
        if source in self.routes:
            # we are not popping the key to prevent calling self.app.add_api_route multiple times
            # when self.add_route is called on the same source_path after removal
            self.routes[source] = None

    def _start(self):
        self._in_subprocess = True
        self.app = FastAPI()
        if self._default_target:
            self._default_session = httpx.AsyncClient(timeout=None)
            self._create_default_route()
        for route in self.routes.values():
            route.start_endpoint_telemetry()
        threading.Thread(target=self._rpc_manager, daemon=True).start()
        uvicorn.run(self.app, port=self.port, host="0.0.0.0", workers=self.workers)

    def _rpc_manager(self):
        while True:
            message = self.message_queue.get()
            if message["method"] == "add_route":
                self.add_route(**message["kwargs"])
            elif message["method"] == "remove_route":
                self.remove_route(**message["kwargs"])

    def start(self):
        self.uvicorn_subprocess = Process(target=self._start)
        self.uvicorn_subprocess.start()

    def stop(self):
        if self.uvicorn_subprocess:
            self.uvicorn_subprocess.terminate()
            self.uvicorn_subprocess = None
