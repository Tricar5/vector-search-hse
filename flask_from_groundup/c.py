import pickle
import cv2
with open('all_data.pickle', 'rb') as handle:
    all_data = pickle.load(handle)
dataset, meta = all_data
all_videos = sorted(set([m[0] for m in meta]))

def open_and_load_frame(meta_, thumbnail_size=240):
    file_name, frame_number = meta_
    if frame_number == -1:
        return cv2.imread(file_name), -1

    cap = cv2.VideoCapture(file_name)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))  
    ret, frame = cap.read()
    cap.release()

    original_height, original_width, _ = frame.shape
    aspect_ratio = thumbnail_size / max(original_height, original_width)
    new_width, new_height = original_width*aspect_ratio, original_height*aspect_ratio
    frame = cv2.resize(frame, (int(new_width), int(new_height)))

    return (frame, fps)

thumbnails_meta = {}
for i,video in enumerate(all_videos):
    frame_number = 0 if video[-4:] not in ['.png','.jpg','webp','jpeg'] else -1
    frame, fps = open_and_load_frame((video,frame_number))
    thumbnails_meta[video] = [i,fps]
    # cv2.imwrite(f'./thumbnails/{i:06}.jpg', frame)

with open('thumbnails_meta.pickle', 'wb') as handle:
    pickle.dump(thumbnails_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
