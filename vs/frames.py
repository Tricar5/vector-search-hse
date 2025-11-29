import pathlib
import logging

from PIL import Image
from typing import Tuple, List, Generator

import cv2

logger = logging.getLogger(__name__)


def iter_video_frames(
    file_path: str,
    rate: float,
) -> Generator[Image, None, None]:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Cannot open file: {file_path}")
        exit()

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set frames per 1 sec, 2 sec etc.
    frame_rate = fps * rate
    pos = 0
    try:
        while True:
            ret, frame = cap.read()
            pos += 1

            if not ret:
                cap.release()
                break

            # sample one frame per second
            if pos % frame_rate != 0:
                continue
            # convert BGR→RGB and preprocess
            frame = frame[:, :, ::-1]
            img = Image.fromarray(frame)

            yield img, pos
    finally:
        # When everything done, release the capture
        cap.release()


def extract_frame_by_its_pos(
    file_path: str,
    pos: int,
) -> Image:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error('Cannot open file: {file_path}')
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if pos > total_frames:
        raise ValueError
    i = 0
    while True:
        ret, frame = cap.read()
        i += 1
        if not ret:
            break

        # sample one frame per second
        if i == pos:
            frame = frame[:, :, ::-1]
            img = Image.fromarray(frame)
            cap.release()
            return img
    cap.release()


def make_frames_from_video(
    file_path: str,
    save_folder: pathlib.Path,
    rate: float = 1
) -> Tuple[List[int], List[str]]:
    cap = cv2.VideoCapture(file_path)

    frames_path = []
    frames_points = []

    if not cap.isOpened():
        logger.error(f'Cannot open file: %s', file_path)
        exit()
    i = 0
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # CAP_rate = FPS * rate (знаменатель извлечения в секундах )
    cap_rate = (fps * rate)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        if i % cap_rate == 0:
            frame_point = i // fps
            frame_filename = f'{i + 1 // fps:06}.png'
            frame_filepath = str(save_folder / frame_filename)
            cv2.imwrite(frame_filepath, frame)
            frames_path.append(frame_filepath)
            frames_points.append(frame_point)

        i += 1

    # When everything done, release the capture
    cap.release()
    return frames_points, frames_path
