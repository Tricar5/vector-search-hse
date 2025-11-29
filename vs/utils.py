import logging

import pathlib
import glob
from typing import Optional, List

import torch

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def find_all_files_with_pattern(
    pattern: str,
    folder: Optional[pathlib.Path] = None,
) -> List[str]:
    file_path = pattern
    if folder:
        file_path = str(folder / pattern)
    files = glob.glob(file_path)
    print(f'Found {len(files)} files with format {pattern}.')
    return files
