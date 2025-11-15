import uuid
import os
from typing import Optional
import pathlib

from vs.schemas import VideoInfoSchema, ImageInfoSchema
from vs.utils import make_frames_from_video


class VideoFilesProcessor:

    def __init__(
        self,
        base_dir: pathlib.Path,
        img_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.base_dir = base_dir
        self.img_dir = img_dir if img_dir else self.base_dir / 'images'
        self.root_path = pathlib.Path(os.getcwd())

    def process_single_video_file(
        self,
        video_path: pathlib.Path,
    ) -> VideoInfoSchema:
        video_id = uuid.uuid4().hex

        image_dir = self.img_dir / video_id

        if os.path.exists(image_dir):
            os.rmdir(image_dir)
        os.makedirs(self.root_path / image_dir)

        img_paths = make_frames_from_video(
            video_path,
            image_dir,
        )

        imgs = [
            ImageInfoSchema(
                id=uuid.uuid4().hex,
                path=path,
            ) for path in img_paths
        ]

        return VideoInfoSchema(
            id=video_id,
            video_path=os.path.relpath(video_path, os.getcwd()),
            images=imgs,
        )
