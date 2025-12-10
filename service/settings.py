from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        arbitrary_types_allowed=True,
        extra='ignore',
    )

    APP_NAME: str = 'VectorSearchApplication'
    API_PORT: int = 8000
    ENGINE_CONFIG_PATH: str = 'engines.yml'
    POSTGRES_DSN: str


@lru_cache(maxsize=1)
def get_settings(env_file: str = '.env') -> AppSettings:
    """Settings Factory"""
    return AppSettings(_env_file=env_file)


settings = get_settings(env_file='.env')
