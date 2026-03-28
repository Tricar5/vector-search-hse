from typing import List

import clip
import torch
from numpy._typing import NDArray
from abc import ABC, abstractmethod
from PIL import Image
from AudioCLIP.model import AudioClip
from AudioCLIP.utils.transforms import ToTensor1D
import torchvision.transforms as TT
import librosa

# class ClipEmbedder:

#     def __init__(
#         self,
#         batch_size: int,
#     ) -> None:
#         self.model, self.preprocessor = clip.load('ViT-B/32')
#         self.batch_size = batch_size

#     def encode_images(
#         self,
#         batch_tensor: torch.Tensor,
#     ) -> torch.Tensor:
#         with torch.no_grad():
#             embeds = self.model.encode_image(batch_tensor)
#         return embeds

#     def embed_images(
#         self,
#         img_batch: List[NDArray],
#     ) -> List[NDArray]:
#         images = [self.preprocess(img) for img in img_batch]

#         embeddings = []
#         batch_tensor = torch.cat(images, dim=0)
#         for batch in torch.split(batch_tensor, self.batch_size, dim=0):
#             embedded = self.encode_images(batch)
#             for emb in embedded:
#                 emb = self.postprocess(emb)
#                 embeddings.append(emb.detach().numpy())

#         return embeddings

#     def preprocess(
#         self,
#         img: NDArray,
#     ) -> None:
#         return self.preprocessor(img).unsqueeze(0)

#     def postprocess(
#         self,
#         emb: torch.Tensor,
#     ) -> torch.Tensor:
#         return emb / torch.linalg.norm(emb)

class BaseWrapper(ABC):
    def __init__(self, device: str) -> None:
        self.device = device

    def preprocess_image(self, image: Image) -> torch.Tensor:
        raise NotImplementedError("Image processing is not supported by this model")

    def process_image(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Image processing is not supported by this model")

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError("Text processing is not supported by this model")

    def process_text(self, batch) -> torch.Tensor:
        raise NotImplementedError("Text processing is not supported by this model")

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        raise NotImplementedError("Audio processing is not supported by this model")

    def process_audio(self, batch) -> torch.Tensor:
        raise NotImplementedError("Audio processing is not supported by this model")

class CLIPWrapper(BaseWrapper):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.model, self.preprocessor = clip.load('ViT-B/32').to(self.device)
        self.max_tokens = 77  # для CLIP 77 токенов - максимум

    def preprocess_image(self, image: Image) -> torch.Tensor:
        return self.preprocessor(image).unsqueeze(0).to(self.device)

    def process_image(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.model.encode_image(batch)
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def preprocess_text(self, text: str) -> torch.Tensor:
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            # токенизируем слово и проверяем, сколько токенов добавится
            word_tokens = clip._tokenizer.encode(word)
            if current_len + len(word_tokens) <= self.max_tokens:
                current_chunk.append(word)
                current_len += len(word_tokens)
            else:
                # сохраняем текущий чанк и начинаем новый
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_len = len(word_tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return clip.tokenize(chunks).to(self.device)

    def process_text(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.model.encode_text(batch)
        return embeds / embeds.norm(dim=-1, keepdim=True)

class AudioCLIPWrapper(BaseWrapper):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.aclp = AudioCLIP(pretrained='AudioClip/assets/AudioCLIP-Full-Training.pt').to(self.device)
        self.image_size = 224
        self.image_mean = (0.48145466, 0.4578275, 0.40821073)
        self.image_std = (0.26862954, 0.26130258, 0.27577711)
        self.sample_rate = 44100
        self.audio_transforms = ToTensor1D()
        self.image_transforms = TT.Compose([
            TT.ToTensor(),
            TT.Resize(self.image_size, interpolation=Image.BICUBIC),
            TT.CenterCrop(self.image_size),
            TT.Normalize(self.image_mean, self.image_std)
        ])

    def preprocess_image(self, image: Image) -> torch.Tensor:
        return self.image_transforms(image).unsqueeze(0).to(self.device)

    def process_image(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.aclp.encode_image(audio)
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def preprocess_text(self, text: str) -> torch.Tensor:
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            # токенизируем слово и проверяем, сколько токенов добавится
            word_tokens = clip._tokenizer.encode(word)
            if current_len + len(word_tokens) <= 77: # для CLIP 77 токенов - максимум
                current_chunk.append(word)
                current_len += len(word_tokens)
            else:
                # сохраняем текущий чанк и начинаем новый
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_len = len(word_tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return clip.tokenize(chunks).to(self.device)

    def process_text(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.aclp.encode_text(batch)
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        track, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        return self.audio_transform(track.reshape(1,-1)).unsqueeze(0).to(self.device)

    def process_audio(self, batch) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.aclp.audio(batch)
        return embeds / embeds.norm(dim=-1, keepdim=True)

