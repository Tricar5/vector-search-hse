import uuid

from pydantic import (
    BaseModel,
    Field,
)
from typing import List


class BaseInfoClass(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class FrameInfo(BaseInfoClass):
    time: float
    path: str
    video_id: str


class VideoInfo(BaseInfoClass):
    video_path: str
    frames: List[FrameInfo] = Field(default_factory=list)
