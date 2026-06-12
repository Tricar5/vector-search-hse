from typing import Literal

from pydantic import BaseModel


class UsedVideo(BaseModel):
    start_pos: float
    end_pos: float
    score: float


class ParametersSettings(BaseModel):
    frame_threshold: float = 0.26
    percentile: float = 0.8
    video_threshold: float = 0.5


class LocalEngineConfig(BaseModel):
    device: str = 'cpu'
    model_type: Literal['clip', 'audioclip'] = 'clip'
    index_path: str = 'data/index.pkl'
    metadata_path: str = 'data/metadata.pkl'
    thumbnail_path: str = 'data/thumbnails.pkl'
    audio_model_path: str | None = None  # path to AudioCLIP weights, required if model_type='audioclip'
    reranker_path: str | None = None
    image: ParametersSettings
    text: ParametersSettings
    audio: ParametersSettings = ParametersSettings(
        frame_threshold=0.8,
        video_threshold=0.01,
        percentile=0.9,
    )
