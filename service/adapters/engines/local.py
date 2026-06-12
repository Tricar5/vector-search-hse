from vs.embedder.clip import AudioCLIPWrapper, CLIPWrapper
from vs.local.engine import LocalSearchEngine as _VSLocalSearchEngine
from vs.reranker import Reranker

from service.adapters.engines.base import Engine
from service.adapters.engines.schemas import LocalEngineConfig
from service.adapters.files import load_yml_config
from service.domain.videos.schemas import VideoDescription
from service.settings import AppSettings


def _to_service_schema(v: object) -> VideoDescription:
    return VideoDescription(
        name=v.name,  # type: ignore[attr-defined]
        path=v.path,  # type: ignore[attr-defined]
        video_id=v.video_id,  # type: ignore[attr-defined]
        frame_num=v.frame_num,  # type: ignore[attr-defined]
        frame_num_end=v.frame_num_end,  # type: ignore[attr-defined]
        fps=v.fps,  # type: ignore[attr-defined]
        start_pos=v.start_pos,  # type: ignore[attr-defined]
        end_pos=v.end_pos,  # type: ignore[attr-defined]
        score=v.score,  # type: ignore[attr-defined]
    )


class LocalSearchEngine(Engine):
    """
    Thin service adapter around vs.local.LocalSearchEngine.
    Converts vs.local.VideoDescription → service domain VideoDescription.
    """

    def __init__(self, engine: _VSLocalSearchEngine, config: LocalEngineConfig) -> None:
        self._engine = engine
        self._config = config
        # expose for web route /image lookup
        self._int_to_video = engine._int_to_video

    def search_videos_by_text(self, text: str) -> list[VideoDescription]:
        cfg = self._config.text
        results = self._engine.search_by_text(
            text,
            frame_threshold=cfg.frame_threshold,
            video_threshold=cfg.video_threshold,
            percentile=cfg.percentile,
        )
        return [_to_service_schema(v) for v in results]

    def search_videos_by_image(self, img: object) -> list[VideoDescription]:
        cfg = self._config.image
        results = self._engine.search_by_image(
            img,
            frame_threshold=cfg.frame_threshold,
            video_threshold=cfg.video_threshold,
            percentile=cfg.percentile,
        )
        return [_to_service_schema(v) for v in results]

    def search_videos_by_audio(self, audio_path: str) -> list[VideoDescription]:
        cfg = self._config.audio
        results = self._engine.search_by_audio(
            audio_path,
            frame_threshold=cfg.frame_threshold,
            video_threshold=cfg.video_threshold,
            percentile=cfg.percentile,
        )
        return [_to_service_schema(v) for v in results]

    @classmethod
    def build_engine(cls, settings: AppSettings) -> 'LocalSearchEngine':
        raw_config = load_yml_config(settings.engine_config_path)
        config = LocalEngineConfig.parse_obj(raw_config['local'])

        reranker = Reranker(config.reranker_path) if config.reranker_path else None

        if config.model_type == 'audioclip':
            model = AudioCLIPWrapper(device=config.device)
        else:
            model = CLIPWrapper(device=config.device)

        engine = _VSLocalSearchEngine.from_pickle(
            index_path=config.index_path,
            metadata_path=config.metadata_path,
            thumbnail_path=config.thumbnail_path,
            device=config.device,
            model=model,
            reranker=reranker,
        )
        return cls(engine, config)
