from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    port: int = 8000
    index_path: str = 'index.pkl'
    metadata_path: str = 'metadata.pkl'
    thumbnail_path: str = 'thumbnails.pkl'


config = AppConfig()
