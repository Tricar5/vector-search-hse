import logging
import os

import pathlib
import glob
from typing import Optional, List, Set, Union

import torch

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'




def find_files_by_extensions(
    folder: Union[pathlib.Path, str],
    extensions: List[str],
) -> List[str]:
    """Поиск файлов по паттернам"""
    found_files = []
    patterns = ['.' + ext if not ext.startswith('.') else ext for ext in extensions]
    patterns = {pattern.lower() for pattern in patterns}

    # Проходим по всем файлам в директории
    for filename in os.listdir(folder):
        # Получаем расширение файла в нижнем регистре
        file_ext = os.path.splitext(filename)[1].lower()

        # Проверяем, соответствует ли расширение любому из паттернов (в нижнем регистре)
        if file_ext in patterns:
            full_path = os.path.join(folder, filename)
            found_files.append(full_path)

    return found_files


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

