from typing import Optional, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import QueryResponse
from fastembed import ImageEmbedding, TextEmbedding


class QdrantSearchEngine:
    def __init__(
        self,
        vector_client: QdrantClient,
    ) -> None:
        self._vectors = vector_client
        self._txt = TextEmbedding("Qdrant/clip-ViT-B-32-text")
        self._img = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")

    def get_text_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        return list(self._txt.embed(text))[0]

    def search_by_text(
        self,
        text: str,
        limit: Optional[int] = 10,
    ) -> Any:
        text_embedding = self.get_text_embedding(text)
        query_response = self._search(text_embedding, limit)

        return query_response.points

    def _search(
        self,
        query: np.ndarray,
        limit: Optional[int] = 10,
    ) -> QueryResponse:
        return self._vectors.query_points(
            query=query,
            limit=limit,
        )
