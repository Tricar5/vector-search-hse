import cv2
import pathlib
import glob
from typing import Optional


def make_frames_from_video(
    file_path: str,
    save_folder: pathlib.Path,
) -> list[str]:
    cap = cv2.VideoCapture(file_path)

    frames_path = []

    if not cap.isOpened():
        print(f"Cannot open file: {file_path}")
        exit()
    i = 0
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if (i % fps * 2 == 0):
            frame_filename = f'{i + 1 // fps:06}.png'
            frame_filepath = str(save_folder / frame_filename)
            cv2.imwrite(frame_filepath, frame)
            frames_path.append(frame_filepath)

        i += 1

    # When everything done, release the capture
    cap.release()
    return frames_path


def find_all_files_with_pattern(
    pattern: str,
    folder: Optional[pathlib.Path] = None,
) -> list[str]:
    file_path = pattern
    if folder:
        file_path = str(folder / pattern)
    files = glob.glob(file_path)
    print(f'Found {len(files)} files with format {pattern}.')
    return files
