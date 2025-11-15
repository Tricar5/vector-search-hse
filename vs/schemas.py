from pydantic import BaseModel
from typing import List


class ImageInfoSchema(BaseModel):
    id: str
    path: str


class VideoInfoSchema(BaseModel):
    id: str
    video_path: str
    images: List[ImageInfoSchema]
