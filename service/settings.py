from dotenv import (
    find_dotenv,
    load_dotenv,
)
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra='ignore',
    )


class AuthConfig(_Settings):
    auth_public_key: str
    auth_private_key: str


class DbConfig(_Settings):
    db_dsn: str
    min_db_pool_size: int = 2
    max_db_pool_size: int = 5

    @property
    def async_db_dsn(self) -> str:
        return self.db_dsn.replace('postgresql://', 'postgresql+asyncpg://')


class AppSettings(_Settings):
    name: str  # = 'VectorSearchApp'
    engine_config_path: str  # = 'engines.yml'
    auth: AuthConfig = Field(default_factory=AuthConfig)
    db: DbConfig = Field(default_factory=DbConfig)


def get_settings(env_file: str = '.env') -> AppSettings:
    """Settings Factory"""
    load_dotenv(find_dotenv(env_file))
    return AppSettings()


settings = get_settings()
