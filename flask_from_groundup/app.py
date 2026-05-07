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
from vs.embedder.clip import AudioCLIPWrapper, CLIPWrapper
from collections import Counter
import pandas as pd
app = Flask(__name__)
with open('thumbnails_meta.pickle', 'rb') as handle:
    thumbnails_meta = pickle.load(handle)
# with open('index.pkl', 'rb') as handle:
#     dataset = pickle.load(handle)
# with open('metadata.pkl', 'rb') as handle:
#     meta = pickle.load(handle)
with open('all_data.pickle', 'rb') as handle:
    dataset, meta = pickle.load(handle)
dataset = torch.tensor(np.array(dataset)).cuda()
# dataset /= torch.linalg.norm(dataset, dim=-1, keepdim=True)
with open('model.pkl', 'rb') as handle:
    reranker = pickle.load(handle)
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# aclp = AudioCLIPWrapper(device)
aclp = CLIPWrapper(device)

meta_total_frames = Counter([m[0] for m in meta])
# print(meta_total_frames)
all_videos = sorted(set([m[0] for m in meta]))
video_to_int = {v: i for i, v in enumerate(all_videos)}
int_to_video = {i: v for v, i in video_to_int.items()}

# Формируем мета как GPU массивы
meta_video_ids = torch.tensor([video_to_int[m[0]] for m in meta], dtype=torch.int32)
meta_frame_nums_s = torch.tensor([m[1] for m in meta], dtype=torch.int32)
meta_frame_nums_e = torch.tensor([m[2] for m in meta], dtype=torch.int32)

def brute_force_query_torch(X, x, certainty_threshold):
    print(x.shape, X.t().shape)
    sims = (x @ X.t())[0]   # shape: [N]
#    print(sims)

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
            data = aclp.process_image(aclp.preprocess_image(file))#.squeeze()
    if len(data) == 0:
        return redirect('/')
    # data /= torch.linalg.norm(data)
    # data = torch.sign(data)*torch.pow(torch.abs(data),0.25)
    # data /= torch.linalg.norm(data)
    return render_main_page(data, video_threshold=0.00, frame_threshold=0.15, percentile=0.9)
    # return render_main_page(data, video_threshold=0.65, frame_threshold=0.4, percentile=1)
    # return render_main_page(data, video_threshold=0.72, frame_threshold=0.6, percentile=0.80)
    # return render_main_page(data, video_threshold=0.6, frame_threshold=0.65, percentile=0.85)
import tempfile
@app.route('/load_audio', methods=['POST', 'GET'])
def upload_audio():
    if request.method == 'GET':
        return redirect('/')
    file = request.files['file']
    r = int(request.form['start_pos'])
    data = []
    if file.filename != '':
        with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        batch, _ = aclp.preprocess_audio(temp_path)
        embedding = aclp.process_audio(batch)
        print(data)
        data = embedding[0].unsqueeze(0)
    if len(data) == 0:
        return redirect('/')
    # data = torch.sign(data)*torch.pow(torch.abs(data),0.25)
    # data /= torch.linalg.norm(data)
    # data = data.unsqueeze(0)
    # print(data.shape)
    # return render_main_page(data, video_threshold=0.0, frame_threshold=0.01, percentile=1)
    return render_main_page(data, video_threshold=0.01, frame_threshold=0.8, percentile=0.9)
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
            data = aclp.process_text(aclp.preprocess_text(text))
    if len(data) == 0:
        print('lol')
        return redirect('/')
    # data /= torch.linalg.norm(data)
    return render_main_page(data, video_threshold=0.1, frame_threshold=0.05, percentile=0.9, orig_data=text)
    # return render_main_page(data, video_threshold=0.0, frame_threshold=0.12, percentile=1)
    # return render_main_page(data, video_threshold=0.01, frame_threshold=0.01, percentile=0.85, orig_data=text)
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


def compute_stats(certs, total_frames):
    certs = certs.cpu().detach().numpy() if torch.is_tensor(certs) else certs
    if len(certs) == 0:
        certs = np.array([0.0])
    features = {}
    features['max'] = certs.max().item()
    features['mean'] = certs.mean().item()
    features['std'] = certs.std().item()
    features['perc_90'] = np.percentile(certs, 90).item()
    features['num_passed'] = len(certs)
    print(len(certs), total_frames)
    features['density'] = len(certs) / max(1, total_frames)
    features['range'] = (certs.max() - certs.min()).item()
    return features


def render_main_page(
        data,
        video_threshold=0.8,
        frame_threshold=0.8,
        percentile=0.8,
        orig_data=None):
    s_ = time.time()
    idxs, certs = brute_force_query_torch(dataset, data, frame_threshold)
    certs = certs.cpu()
    idxs = idxs.cpu()
    video_idxs = meta_video_ids[idxs]
    video_frames_s = meta_frame_nums_s[idxs]
    video_frames_e = meta_frame_nums_e[idxs]

    vals, order = torch.sort(video_idxs)
    targets = torch.tensor([video_to_int[v] for v in all_videos])
    left  = torch.bucketize(targets, vals, right=False)
    right = torch.bucketize(targets, vals, right=True)
    
    num_videos = len(all_videos)
    print(certs)
    
    lengths = right - left
    valid = lengths > 0
    print(torch.sum(valid))

    perc_offsets = (lengths.float() * (1 - percentile)).long()
    perc_offsets = torch.clamp(perc_offsets, min=0)

    print(certs)
    perc_idxs = left + perc_offsets
    perc_idxs = perc_idxs[valid]
    certs_per_video = certs[order[perc_idxs]]
    passed = certs_per_video >= video_threshold
    print(torch.sum(passed))

    final_video_idxs = torch.nonzero(valid)[passed].squeeze(1)
    final_certs = certs_per_video[passed]
    frames_sorted_s = video_frames_s[order]
    frames_sorted_e = video_frames_e[order]

    used_videos = {}
    video_descriptions = []
    
    for i, vid_idx in enumerate(final_video_idxs.tolist()):
        l,r  = left[vid_idx], right[vid_idx]
        print(vid_idx, certs[order[l:r]])
        subset_s = frames_sorted_s[l:r]
        subset_e = frames_sorted_e[l:r]
        
        subset_s = video_frames_s[order[l:r]]
        subset_e = video_frames_e[order[l:r]]

        start = subset_s.min()
        end = subset_e.max()
        max_frame = subset_e[0]
        
        video = all_videos[vid_idx]

        stats = compute_stats(certs[order[l:r]], meta_total_frames[video])
        feature_names = ['max', 'mean', 'std', 'perc_90', 'num_passed', 'range']
        features_df = pd.DataFrame([[stats[name] for name in feature_names]], columns=feature_names)
        # Предсказание вероятности
        prob = reranker.predict_proba(features_df)[0][1]

        used_videos[video] = [
            start.item(),
            end.item(),
            prob
            #final_certs[i].item()
        ]

        frame_request = (
            f"/image?video={video_to_int[video]}"
            f"&frame_number={max_frame.item()}"
        )

        video_descriptions.append(
            (
                video, 
                frame_request, 
                thumbnails_meta[video][1], 
                stats
            )
        )

    results = sorted(
        video_descriptions,
        key=lambda x: used_videos[x[0]][2],
        reverse=True
    )[:25]
    
    # # Для каждого видео подготовим признаки, включая idx (позицию в этом списке)
    # reranked = []
    # for idx, (video, frame_request, thumb, stats) in enumerate(results, start=1):
    #     # Признаки в том же порядке, что и при обучении: ['idx','max','mean','std','perc_90','num_passed','range']
    #     # density не используем (он был константой)
    #     features = [
    #         stats['max'],
    #         stats['mean'],
    #         stats['std'],
    #         stats['perc_90'],
    #         stats['num_passed'],
    #         stats['range']
    #     ]
    #     # Предсказание вероятности класса 1 (релевантен)
    #     feature_names = ['max', 'mean', 'std', 'perc_90', 'num_passed', 'range']
    #     features_df = pd.DataFrame([features], columns=feature_names)
    #     prob = reranker.predict_proba(features_df)[0][1]
    #     reranked.append((video, frame_request, thumb, stats, prob))
    #     used_videos[video][2] = prob
    
    # # Сортируем по новой вероятности (по убыванию)
    # reranked.sort(key=lambda x: x[4], reverse=True)
    
    # # Берём топ-25
    # results = reranked[:25]
    # Для совместимости с последующим кодом можно преобразовать в формат как было:
    # results = [(video, frame_request, thumb, stats), ...] без prob
    # results = [(v, fr, th, st) for v, fr, th, st, _ in results]

    print(time.time()-s_)

    return render_template(
        "index.html",
         frames=results,
         used_videos=used_videos,
         orig_data=orig_data
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

@app.route('/download_csv', methods=['POST', 'GET'])
def download_csv():
    l = int(request.form['len'])
    print(request.form.keys())
    s = 'idx,'+','.join(eval(request.form['1_stats']).keys())+',rel\n'
    print(s)
    for i in range(1,l+1):
        s += f'{i},'
        s += ','.join(map(str,eval(request.form[f"{i}_stats"]).values()))
        s += f',{request.form.get(f"{i}_rel", "off")=="on"}\n'
    with open(f'{request.form["orig_data"]}.csv', 'w') as f:
        f.write(s)
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

