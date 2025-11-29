import pathlib
from typing import Optional, Union, List
import orjson

from vs.schemas import VideoInfo, FrameInfo


class LocalVideoInfoStorage:
    def __init__(
        self,
        video_payloads: list[VideoInfo],
    ) -> None:
        self.videos = video_payloads
        self._id_map = {video.id: idx for idx, video in enumerate(video_payloads)}

    def get_video_info_by_id(
        self,
        video_id: str,
    ) -> VideoInfo:
        video_idx = self._id_map[video_id]
        return self.videos[video_idx]

    def get_frames_by_video_id(
        self,
        video_id: str,
    ) -> List[FrameInfo]:
        video = self.get_video_info_by_id(video_id)
        return video.frames

    def dump_to_file(self, file_path: Union[pathlib.Path, str]) -> None:
        jsonable = orjson.dumps(
            [m.model_dump() for m in self.videos],
            option=orjson.OPT_INDENT_2,
        )
        with open(file_path, 'w') as f:
            f.write(jsonable.decode('utf-8'))

    @classmethod
    def from_file(cls, file_path: Union[pathlib.Path, str]) -> 'LocalVideoInfoStorage':
        with open(file_path, 'rb') as f:
            json_bytes = f.read()
            json_array = orjson.loads(json_bytes)
        payloads = [VideoInfo(**js) for js in json_array]
        return cls(video_payloads=payloads)
