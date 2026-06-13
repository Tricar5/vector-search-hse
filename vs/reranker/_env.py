import os
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return here.parents[2]


def load_mlflow_env() -> str:
    root = _find_project_root()

    from dotenv import load_dotenv

    # .env.default is always loaded first so .env can override it
    load_dotenv(root / '.env.default', override=False)
    load_dotenv(root / '.env', override=True)

    tracking_uri = os.environ.get('MLFLOW_URI', 'http://localhost:5050')

    # Propagate the canonical MLFLOW_TRACKING_URI that the SDK reads
    os.environ.setdefault('MLFLOW_TRACKING_URI', tracking_uri)
    return tracking_uri
