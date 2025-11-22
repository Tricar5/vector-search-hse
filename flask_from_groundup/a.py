import os
import cv2
import tqdm
import torch
import clip
import pickle
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
all_data = {}
folder1 = './zip_to_yandex/'
tbar = tqdm.tqdm([f'{folder1}{item}' for item in os.listdir(folder1)])

def get_images_in_batches(file_path, batch_size=32):
    images_embeds = []
    images_meta = []
    if file_path[-4:] in ['.jpg','.png', 'webp', 'jpeg']:
        img = Image.open(file_path)
        img = preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
                embeds = model.encode_image(img)[0]
        embeds = embeds / torch.linalg.norm(embeds)
        images_embeds.append(embeds.cpu().numpy())
        images_meta.append((file_path, -1))
        return images_embeds, images_meta

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Cannot open file: {file_path}")
        exit()

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch_frames = []       # preprocessed tensors
    batch_metadata = []     # tuples (file_path, frame_index)

    i = -1
    while True:
        ret, frame = cap.read()
        i += 1

        if not ret:
            break

        # sample one frame per second
        if i % fps != 0:
            continue

        tbar.set_description(f'{round(i / total_frames * 100)}%')

        # convert BGR→RGB and preprocess
        frame = frame[:, :, ::-1]
        img = Image.fromarray(frame)
        img = preprocess(img).unsqueeze(0).cuda()

        batch_frames.append(img)
        batch_metadata.append((file_path, i))

        # ---- process full batch ----
        if len(batch_frames) == batch_size:
            batch_tensor = torch.cat(batch_frames, dim=0)
            with torch.no_grad():
                embeds = model.encode_image(batch_tensor)

            # normalize & append
            for emb, meta in zip(embeds, batch_metadata):
                emb = emb / torch.linalg.norm(emb)
                images_embeds.append(emb.cpu().numpy())
                images_meta.append(meta)

            batch_frames = []
            batch_metadata = []

    # ---- flush remaining frames ----
    if batch_frames:
        batch_tensor = torch.cat(batch_frames, dim=0)
        with torch.no_grad():
            embeds = model.encode_image(batch_tensor)

        for emb, meta in zip(embeds, batch_metadata):
            emb = emb / torch.linalg.norm(emb)
            images_embeds.append(emb.cpu().numpy())
            images_meta.append(meta)

    cap.release()
    return images_embeds, images_meta


all_embeds = []
all_meta = []
for f in tbar:
    # print(f)
    images_embeds, images_meta = get_images_in_batches(f, batch_size=128)
    all_embeds.extend(images_embeds)
    all_meta.extend(images_meta)

with open('all_data.pickle', 'wb') as handle:
    pickle.dump((all_embeds, all_meta), handle, protocol=pickle.HIGHEST_PROTOCOL)
