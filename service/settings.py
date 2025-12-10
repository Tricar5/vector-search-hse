from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    port: int = 8000
    index_path: str = 'data/index.pkl'
    metadata_path: str = 'data/metadata.pkl'
    thumbnail_path: str = 'data/thumbnails.pkl'


config = AppConfig()
