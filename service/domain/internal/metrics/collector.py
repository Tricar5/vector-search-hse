from typing import (
    Any,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

import re


def _to_snakecase(name: str) -> str:
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class MetricsCollector:  # noqa: WPS214, WPS338
    __instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> 'MetricsCollector':
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)  # noqa: WPS601
            cls.__instance._metrics = {}  # noqa: WPS437
        return cls.__instance

    _app_name: str = ''

    def __init__(self) -> None:
        self._metrics: Dict[str, Any]

    @classmethod
    def set_app_name(cls, name: str) -> None:  # noqa: WPS615
        cls._app_name = _to_snakecase(name)  # noqa: WPS601

    def observe_search_query(self, query_type: str, text: str) -> None:
        self.hist(
            name='search_text_query_length',
            docs='Length of text search queries in characters',
            labels=('query_type',),
            buckets=(5, 10, 20, 50, 100, 200),
        ).labels(query_type=query_type).observe(len(text))
        self.hist(
            name='search_text_query_tokens',
            docs='Number of tokens in text search queries',
            labels=('query_type',),
            buckets=(1, 5, 10, 20, 50),
        ).labels(query_type=query_type).observe(len(text.split()))

    def observe_search_duration(self, query_type: str, duration: float) -> None:
        self.hist(
            name='search_duration',
            docs='Time spent processing search queries',
            labels=('query_type',),
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        ).labels(query_type=query_type).observe(duration)

    def observe_search_results(self, query_type: str, n_results: int) -> None:
        self.hist(
            name='search_results_count',
            docs='Number of results returned per search query',
            labels=('query_type',),
            buckets=(0, 1, 3, 5, 10, 15, 25),
        ).labels(query_type=query_type).observe(n_results)

    def db_hist(self, repo: str, method: str) -> ContextManager[Histogram]:
        metric = self.hist(
            name='db_query_duration_seconds',
            docs='Time spent in database queries',
            labels=('repo', 'method'),
        )
        return metric.labels(repo=repo, method=method).time()

    def task_hist(self, strategy: str, method: str) -> ContextManager[Histogram]:
        metric = self.hist(
            name='task_duration',
            docs='Time spent processing task',
            labels=('name', 'method'),
        )
        return metric.labels(name=strategy, method=method).time()

    def client_hist(
        self,
        service: str,
        method: str,
        name: str,
    ) -> ContextManager[Histogram]:
        metric = self.hist(
            name='http_client_query_duration_seconds',
            docs='Time spent in http clients queries',
            labels=('service', 'method', 'name'),
        )
        return metric.labels(service=service, method=method, name=name).time()

    def client_counter(self, service: str, method: str, name: str, status: int) -> None:
        metric = self.counter(
            name='http_client_status_count',
            docs='HTTP client status counter',
            labels=('service', 'method', 'name', 'status'),
        )
        metric.labels(service=service, method=method, name=name, status=status).inc()

    def status_counter(self, method: str, name: str, status: int) -> None:
        metric = self.counter(
            name='status_request_count',
            docs='HTTP status counter',
            labels=('method', 'name', 'status'),
        )
        metric.labels(method=method, name=name, status=status).inc()

    def error_code_counter(self, method: str, name: str, error_code: str) -> None:
        metric = self.counter(
            name='error_code_request_count',
            docs='Exception error code counter',
            labels=('method', 'name', 'error_code'),
        )
        metric.labels(method=method, name=name, error_code=error_code).inc()

    def count_handler_statuses(self, handler_name: str, status: int) -> None:
        counter = self.counter(
            name=f'{handler_name.lower()}_status_request_count',
            docs='HTTP status counter',
            labels=['status'],
        )
        counter.labels(status=status).inc()

    def count_handler_errors(self, handler_name: str, error_code: str) -> None:
        metric = self.counter(
            name=f'{handler_name.lower()}_error_code_request_count',
            docs='Exception error code counter',
            labels=['error_code'],
        )
        metric.labels(error_code=error_code).inc()

    def hist(
        self,
        name: str,
        docs: str,
        labels: Iterable[str],
        buckets: Tuple[float, ...] = Histogram.DEFAULT_BUCKETS,
    ) -> Histogram:
        metric = self._metrics.get(name)
        if not metric:
            metric = Histogram(
                name=name,
                documentation=docs,
                labelnames=labels,
                namespace=self._app_name,
                unit='seconds',
                buckets=buckets,
            )
            self._metrics[name] = metric
        return self._metrics[name]

    def counter(self, name: str, docs: str, labels: Iterable[str]) -> Counter:
        metric = self._metrics.get(name)
        if not metric:
            metric = Counter(
                name=name,
                documentation=docs,
                namespace=self._app_name,
                labelnames=labels,
            )
            self._metrics[name] = metric
        return metric

    def gauge(self, name: str, docs: str, labels: Optional[List[str]] = None) -> Gauge:
        metric = self._metrics.get(name)
        if not metric:
            metric = Gauge(
                name=name,
                documentation=docs,
                namespace=self._app_name,
                labelnames=labels or [],
            )
            self._metrics[name] = metric
        return metric

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics
