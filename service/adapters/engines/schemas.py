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
    index_path: str = 'data/index.pkl'
    metadata_path: str = 'data/metadata.pkl'
    thumbnail_path: str = 'data/thumbnails.pkl.'
    image: ParametersSettings
    text: ParametersSettings
