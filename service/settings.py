from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        arbitrary_types_allowed=True,
        extra='ignore',
    )

    APP_NAME: str = 'VectorSearchApp'
    API_PORT: int = 8000
    ENGINE_CONFIG_PATH: str = 'engines.yml'
    POSTGRES_DSN: str


def get_settings() -> AppSettings:
    """Settings Factory"""
    return AppSettings()


settings = get_settings()
