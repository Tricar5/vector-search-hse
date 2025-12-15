from typing import (
    Callable,
    List,
    Optional,
    Union,
)

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import (
    Instrumentator,
    metrics,
)


class MetricsInstrumentator:
    def __init__(self, instrumentator: Optional[Instrumentator] = None) -> None:
        if instrumentator is None:
            instrumentator = Instrumentator(should_group_status_codes=False)
        self.instrumentator = instrumentator

    def setup(self, app: FastAPI, app_name: str) -> None:
        default_metrics = self._set_default_metrics(app_name=app_name)
        for metric in default_metrics:
            self.instrumentator.add(metric)
        self.instrumentator.instrument(app=app).expose(app=app, endpoint='/stats')

    def _set_default_metrics(
        self,
        app_name: str,
    ) -> List[Union[Callable[[metrics.Info], None], None]]:
        return [
            metrics.latency(
                metric_name='api_request_duration_seconds',
                metric_namespace=app_name,
                should_include_method=True,
                should_include_status=True,
                buckets=(
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ),
            ),
            metrics.requests(
                metric_name='status_request_count',
                metric_namespace=app_name,
                should_include_method=True,
                should_include_status=True,
            ),
        ]
