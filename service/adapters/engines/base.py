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
    """
    Base interface for local video search engines.

    Defines public API for:
    - text search
    - image search
    """

    # === Public API ===

    @abstractmethod
    def search_videos_by_text(self, text: str) -> list[VideoDescription]:
        """
        Search videos by text query.
        """
        raise NotImplementedError

    @abstractmethod
    def search_videos_by_image(
        self,
        img: FloatArray,
    ) -> list[VideoDescription]:
        """
        Search videos by image.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_engine(
        cls,
        settings: AppSettings,
    ) -> "Engine":
        """
        Load engine with index and metadata from disk.
        """
        raise NotImplementedError
