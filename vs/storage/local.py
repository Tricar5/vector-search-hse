import pickle
from typing import Any


class LocalIndexStorage:

    def __init__(self, index_path: str, metadata_path: str) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path

    def save_index(self, index: Any) -> None:
        with open(self.index_path, 'wb') as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_metadata(self, metadata: Any) -> None:
        with open(self.metadata_path, 'wb') as file:
            pickle.dump(metadata, file, protocol=pickle.HIGHEST_PROTOCOL)
