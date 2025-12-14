from prometheus_client import Histogram

from service.settings import settings


text_length_metric = Histogram(
    namespace=settings.app_name,
    name='search_text_query_length',
    documentation='Length of text search queries in characters',
    buckets=(
        5,
        10,
        20,
        50,
        100,
    ),
)

token_count_metric = Histogram(
    namespace=settings.app_name,
    name='search_text_query_tokens',
    documentation='Number of tokens in text search queries',
    buckets=(
        5,
        10,
        20,
    ),
)
