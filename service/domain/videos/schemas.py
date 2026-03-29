from pydantic import BaseModel


class VideoDescription(BaseModel):
    name: str
    path: str
    video_id: int
    frame_num: int
    fps: int
    start_pos: float
    end_pos: float
    score: float
