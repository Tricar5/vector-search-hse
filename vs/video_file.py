import uuid
import os
from typing import Optional, List, Union, Generator
import pathlib

from vs.schemas import VideoInfo, FrameInfo
from vs.frames import make_frames_from_video


class VideoFilesProcessor:
    def __init__(
        self,
        base_dir: Optional[pathlib.Path] = None,
        frame_dir: Optional[pathlib.Path] = None,
        frame_rate: float = 1.0
    ) -> None:
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.img_dir = frame_dir if frame_dir else self.base_dir / 'frames'
        self.frame_rate = frame_rate

    def _register_new_video(
        self,
        video_path: Union[str, pathlib.Path],
    ) -> VideoInfo:
        return VideoInfo(video_path=os.path.relpath(video_path, self.base_dir))

    def process_video_generator(
        self,
        video_files: List[Union[pathlib.Path, str]],
    ) -> Generator[VideoInfo, None, None]:
        for video_path in video_files:
            yield self.process_single_video(video_path)

    def process_videos(
        self,
        video_files: List[Union[pathlib.Path, str]],
    ) -> List[VideoInfo]:

        videos = []
        for video_path in video_files:
            video = self.process_single_video(video_path)
            videos.append(video)

        return videos

    def process_single_video(self, video_path: Union[pathlib.Path, str]) -> VideoInfo:
        video = self._register_new_video(video_path)
        frames = self.slice_single_video_file(
            video=video,
        )
        video.frames = frames
        return video

    def slice_single_video_file(
        self,
        video: VideoInfo,
    ) -> List[FrameInfo]:
        # Делаем путь куда будем сохранять фреймы
        image_dir = self.img_dir / video.id

        if os.path.exists(image_dir):
            os.rmdir(image_dir)
        os.makedirs(self.base_dir / image_dir)
        # Получаем time позиции и пути до фреймов
        frame_times, frames_path = make_frames_from_video(
            video.video_path,
            image_dir,
            rate=self.frame_rate,
        )
        # Оборачиваем полученные
        return self._register(frames_path, frame_times, video.id)

    def _register(
        self,
        img_paths: List[str],
        frame_times: Optional[List[int]] = None,
        video_id: Optional[str] = None,
    ) -> List[FrameInfo]:
        """Метод для конвертирования фреймов"""
        return [
            FrameInfo(
                id=uuid.uuid4().hex,
                time=t,
                path=path,
                video_id=video_id
            )
            for path, t in zip(img_paths, frame_times)
        ]
