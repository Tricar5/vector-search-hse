from flask import *
import pickle
import torch
import clip
from PIL import Image
import os
import numpy as np
import cv2
import base64
import tqdm
import torch.nn.functional as F
import subprocess
import time
from io import BytesIO
app = Flask(__name__)
with open('all_data.pickle', 'rb') as handle:
    all_data = pickle.load(handle)
with open('thumbnails_meta.pickle', 'rb') as handle:
    thumbnails_meta = pickle.load(handle)
dataset, meta = all_data
dataset = torch.tensor(np.array(dataset)).cuda()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_videos = sorted(set([m[0] for m in meta]))
video_to_int = {v: i for i, v in enumerate(all_videos)}
int_to_video = {i: v for v, i in video_to_int.items()}

# Формируем мета как GPU массивы
meta_video_ids = torch.tensor([video_to_int[m[0]] for m in meta],
                              device="cuda", dtype=torch.int32)
meta_frame_nums = torch.tensor([m[1] for m in meta],
                               device="cuda", dtype=torch.int32)

def brute_force_query_torch(X, x, certainty_threshold):
    sims = (x @ X.t()).squeeze(0)   # shape: [N]

    # Фильтрация по порогу 0.2
    mask = sims >= certainty_threshold
    filtered_indices = torch.nonzero(mask).squeeze(1)   # индексы в X
    filtered_sims = sims[filtered_indices]

    # Сортировка по убыванию
    sorted_sims, order = torch.sort(filtered_sims, descending=True)
    sorted_indices = filtered_indices[order]

    return sorted_indices, sorted_sims.float()

@app.route('/')
def main():
    return render_template("index.html")


@app.route('/load_image', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'GET':
        return redirect('/')
    file = request.files['file']
    data = []
    if file.filename != '':
        file = Image.open(file)
        with torch.no_grad():
            data = model.encode_image(preprocess(file).unsqueeze(0).cuda())#.squeeze()
    if len(data) == 0:
        return redirect('/')
    # data /= torch.linalg.norm(data)
    data = torch.sign(data)*torch.pow(torch.abs(data),0.25)
    data /= torch.linalg.norm(data)
    return render_main_page(data, video_threshold=0.30, frame_threshold=0.2, percentile=1)
    # return render_main_page(data, video_threshold=0.65, frame_threshold=0.4, percentile=1)
    # return render_main_page(data, video_threshold=0.72, frame_threshold=0.6, percentile=0.80)
    # return render_main_page(data, video_threshold=0.6, frame_threshold=0.65, percentile=0.85)


@app.route('/load_text', methods=['POST', 'GET'])
def upload_text():
    if request.method == 'GET':
        return redirect('/')
    text = request.form['text']
    if text != '':
        with torch.no_grad():
            data = model.encode_text(clip.tokenize([text]).cuda())#.squeeze()
    anti = request.form['anti']
    if anti != '':
        with torch.no_grad():
            data -= 0.5 * model.encode_text(clip.tokenize([anti]).cuda())#.squeeze()
    if len(data) == 0:
        print('lol')
        return redirect('/')
    data /= torch.linalg.norm(data)
    # return render_main_page(data, video_threshold=0.26, frame_threshold=0.15, percentile=0.9)
    return render_main_page(data, video_threshold=0.26, frame_threshold=0.2, percentile=0.8)
    # для усреднений 0.25, 0.2, 1 - норм 

def open_and_load_frame(meta_, thumbnail_size=None):
    file_name, frame_number = meta_
    if frame_number == -1:
        frame = cv2.imread(file_name)
    else:
        cap = cv2.VideoCapture(file_name)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))  
        ret, frame = cap.read()
        cap.release()
    if thumbnail_size:
        original_height, original_width, _ = frame.shape
        aspect_ratio = thumbnail_size / max(original_height, original_width)
        new_width, new_height = original_width*aspect_ratio, original_height*aspect_ratio
        frame = cv2.resize(frame, (int(new_width), int(new_height)))

    return frame


def render_main_page(
        data,
        video_threshold=0.8,
        frame_threshold=0.8,
        percentile=0.8,
        max_workers=4):
    s_ = time.time()
    idxs, certs = brute_force_query_torch(dataset, data, frame_threshold)
    certs = certs.cpu()
    video_idxs = meta_video_ids[idxs]
    video_frames = meta_frame_nums[idxs]
    
    video_descriptions = []
    used_videos = {}

    vals, order = torch.sort(video_idxs)
    targets = torch.tensor([video_to_int[v] for v in all_videos],
                       device=video_idxs.device)
    order = order.cpu()
    left  = torch.bucketize(targets, vals, right=False).cpu()
    right = torch.bucketize(targets, vals, right=True).cpu()

    for i, video in enumerate(all_videos): 
        if left[i] == right[i]:
            continue
        args = order[left[i]:right[i]]
        cert_ = certs[order[left[i]+int((right[i]-left[i])*(1-percentile))]]
        if cert_ < video_threshold:
            continue
        subset = video_frames[args]
        start_ = torch.min(subset)
        end_ = torch.max(subset)
        max_frame = subset[0]
        used_videos[video] = (start_.item(), end_.item(), cert_.item())
        frame_request = f'/image?video={video_to_int[video]}&frame_number={max_frame}'
        video_descriptions.append((video,frame_request,thumbnails_meta[video][1]))
    
    if not video_descriptions:
        return render_template("index.html", frames=[], used_videos={})
    print(time.time()-s_)

    results = sorted(video_descriptions, key=lambda x: used_videos[x[0]][2], reverse=True)[:100]

    return render_template(
        "index.html",
        frames=results,
        used_videos=used_videos
    )

@app.route("/image")
def serve_image():
    # Читаем изображение через cv2
    video = int_to_video[int(request.args.get("video"))]
    frame_number = int(request.args.get("frame_number"))
    thumbnail_size = request.args.get("thumbnail_size")
    thumbnail_size = int(thumbnail_size) if thumbnail_size else None
    img = open_and_load_frame((video, frame_number), thumbnail_size)

    # Кодируем в PNG в память
    ok, encoded_img = cv2.imencode(".jpg", img)
    if not ok:
        return "Cannot encode image", 500

    img_bytes = BytesIO(encoded_img.tobytes())

    # Отправляем
    is_thumbnail = '_thumbnail' if thumbnail_size else ''
    is_frame = f'_frame_{frame_number}' if frame_number != -1 else ''
    filename = os.path.basename(video).rsplit('.', 1)[0]
    filename = f"{filename}{is_frame}{is_thumbnail}.jpg"

    resp = send_file(
        img_bytes,
        mimetype="image/jpeg",
        download_name=filename
    )

    # Чтобы браузеры знали имя файла при скачивании
    resp.headers["Content-Disposition"] = f'inline; filename="{filename}"'

    return resp

@app.route("/video_segment")
def video_segment():
    video_path = request.args.get("path")
    fps = int(request.args.get("fps"))
    frame_start = request.args.get("frame_start", type=int)
    frame_end = request.args.get("frame_end", type=int)

    if not video_path or frame_start is None or frame_end is None:
        abort(400, "Missing parameters")

    if not os.path.exists(video_path):
        abort(404, "Video not found")

    start_seconds = frame_start / fps
    end_seconds = frame_end / fps
    duration = end_seconds - start_seconds

    # Команда ffmpeg для резки без tmp
    cmd = [
        "ffmpeg",
        "-ss", str(start_seconds),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "-crf", "29",
        "-"
    ]

    def generate():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for chunk in iter(lambda: p.stdout.read(64 * 1024), b""):
                yield chunk
        finally:
            p.kill()

    # имя файла — динамическое
    filename = os.path.basename(video_path).rsplit('.', 1)[0]
    filename = f"{filename}_segment_{frame_start}-{frame_end}.mp4"

    resp = Response(generate(), mimetype="video/mp4")

    # ВОТ ГЛАВНОЕ:
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    resp.headers["Accept-Ranges"] = "bytes"
    return resp



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

