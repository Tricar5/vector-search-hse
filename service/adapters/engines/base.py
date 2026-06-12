from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

import numpy as np
from numpy.typing import NDArray

from service.domain.videos.schemas import VideoDescription
from service.settings import AppSettings


FloatArray = NDArray[np.floating[Any]]


class Engine(ABC):

    @abstractmethod
    def search_videos_by_text(self, text: str) -> list[VideoDescription]:
        raise NotImplementedError

    @abstractmethod
    def search_videos_by_image(self, img: FloatArray) -> list[VideoDescription]:
        raise NotImplementedError

    @abstractmethod
    def search_videos_by_audio(self, audio_path: str) -> list[VideoDescription]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_engine(cls, settings: AppSettings) -> 'Engine':
        raise NotImplementedError
