from typing import List

from fastembed import ImageEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from vs.schemas import VideoInfo


class QdrantFramesIndexer:
    def __init__(
        self,
        embedding_model: str,
        vectors: QdrantClient,
    ) -> None:
        self._embedder = ImageEmbedding(embedding_model)
        self._vectors = vectors

    def create_collection(
        self,
        collection_name: str,
        recreate: bool = False,
    ) -> None:
        if recreate and not self._vectors.collection_exists(collection_name):
            self._vectors.delete_collection(collection_name)

        if not self._vectors.collection_exists(collection_name):
            self._vectors.create_collection(collection_name)

    def prepare_points(
        self,
        video: VideoInfo,
    ) -> List[PointStruct]:
        image_paths = [img.path for img in video.frames]
        embeddings = self._embedder.embed(image_paths)

        return [
            PointStruct(
                # По id векторы не дублируются
                id=image.id,
                vector=vector,
                payload={
                    'image_path': image.path,
                    'video_path': video.video_path,
                    'video_id': video.id,
                },
            )
            for vector, image in zip(embeddings, video.frames)
        ]

    def index_video(
        self,
        video: VideoInfo,
        collection_name: str,
    ) -> None:
        points = self.prepare_points(video)
        self._vectors.upsert(
            collection_name=collection_name,
            points=points,
        )

    def index_videos(
        self,
        video_infos: List[VideoInfo],
        collection_name: str,
    ) -> None:
        for video_info in video_infos:
            self.index_video(video_info, collection_name)
