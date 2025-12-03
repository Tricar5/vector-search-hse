from flask import *
import torch
from PIL import Image
import os
import cv2
import subprocess
from io import BytesIO

from vs.local.engine import load_search_index
from service.utils import render_main_page
from vs.frames import open_and_load_frame
from service.settings import config

app = Flask(__name__)

search_index = load_search_index(
    index_path=config.index_path,
    metadata_path=config.metadata_path,
    thumbnail_path=config.thumbnail_path,
)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/load_image', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'GET':
        return redirect('/')
    file = request.files['file']
    data = []
    if file.filename != '':
        file = Image.open(file)
    with torch.no_grad():
        data = search_index.encode_image(file)  # .squeeze()
    if len(data) == 0:
        return redirect('/')
    video_descriptions, used_videos = search_index.query_videos_by_tensor(
        data,
        video_threshold=0.30,
        frame_threshold=0.2,
        percentile=1,
    )
    return render_main_page(video_descriptions, used_videos)


@app.route('/load_text', methods=['POST', 'GET'])
def upload_text():
    if request.method == 'GET':
        return redirect('/')
    text = request.form['text']
    if text != '':
        with torch.no_grad():
            data = search_index.encode_text(text)
    if len(data) == 0:
        print('lol')
        return redirect('/')
    video_descriptions, used_videos = search_index.query_videos_by_tensor(
        data,
        video_threshold=0.26,
        frame_threshold=0.2,
        percentile=0.8,
    )
    return render_main_page(video_descriptions, used_videos)


@app.route('/image')
def serve_image():
    # Читаем изображение через cv2
    int_to_video = search_index.int_to_video
    video = int_to_video[int(request.args.get('video'))]
    frame_number = int(request.args.get('frame_number'))
    thumbnail_size = request.args.get('thumbnail_size')
    thumbnail_size = int(thumbnail_size) if thumbnail_size else None
    img, fps = open_and_load_frame((video, frame_number), thumbnail_size)

    # Кодируем в PNG в память
    ok, encoded_img = cv2.imencode('.jpg', img)
    if not ok:
        return 'Cannot encode image', 500

    img_bytes = BytesIO(encoded_img.tobytes())

    # Отправляем
    is_thumbnail = '_thumbnail' if thumbnail_size else ''
    is_frame = f'_frame_{frame_number}' if frame_number != -1 else ''
    filename = os.path.basename(video).rsplit('.', 1)[0]
    filename = f'{filename}{is_frame}{is_thumbnail}.jpg'

    resp = send_file(
        img_bytes,
        mimetype='image/jpeg',
        download_name=filename
    )

    # Чтобы браузеры знали имя файла при скачивании
    resp.headers['Content-Disposition'] = f"inline; filename='{filename}'"

    return resp


@app.route('/video_segment')
def video_segment():
    video_path = request.args.get('path')
    fps = int(request.args.get('fps'))
    frame_start = request.args.get('frame_start', type=int)
    frame_end = request.args.get('frame_end', type=int)

    if not video_path or frame_start is None or frame_end is None:
        abort(400, 'Missing parameters')

    if not os.path.exists(video_path):
        abort(404, 'Video not found')

    start_seconds = frame_start / fps
    end_seconds = frame_end / fps
    duration = end_seconds - start_seconds

    # Команда ffmpeg для резки без tmp
    cmd = [
        'ffmpeg',
        '-ss', str(start_seconds),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-c:a', 'aac',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-f', 'mp4',
        '-crf', '29',
        '-'
    ]

    def generate():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for chunk in iter(lambda: p.stdout.read(64 * 1024), b''):
                yield chunk
        finally:
            p.kill()

    # имя файла — динамическое
    filename = os.path.basename(video_path).rsplit('.', 1)[0]
    filename = f'{filename}_segment_{frame_start}-{frame_end}.mp4'

    resp = Response(generate(), mimetype='video/mp4')

    # ВОТ ГЛАВНОЕ:
    resp.headers['Content-Disposition'] = f"attachment; filename='{filename}'"
    resp.headers['Accept-Ranges'] = 'bytes'
    return resp
