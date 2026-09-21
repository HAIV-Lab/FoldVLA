"""Episode-preserving LeRobot 3.0 reader used by offline Shadow Replay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


REQUIRED_FRAME_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)


@dataclass
class EpisodeData:
    """One complete, ordered demonstration episode."""

    episode_id: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    instructions: List[str]
    optional_fields: Dict[str, Optional[np.ndarray]]
    source_length: int
    frame_stride: int

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def validate(self, expected_action_dim: Optional[int] = None) -> None:
        length = len(self)
        if length == 0:
            raise ValueError(f"Episode {self.episode_id} is empty")
        for name, value in (
            ("frame_indices", self.frame_indices),
            ("timestamps", self.timestamps),
            ("states", self.states),
        ):
            if len(value) != length:
                raise ValueError(
                    f"Episode {self.episode_id}: {name} has {len(value)} rows, expected {length}"
                )
        if len(np.unique(self.frame_indices)) != length:
            raise ValueError(f"Episode {self.episode_id} contains duplicate frame indices")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError(f"Episode {self.episode_id} frames are not strictly ordered")
        if self.actions.ndim != 2:
            raise ValueError(
                f"Episode {self.episode_id}: action must be [T,D], got {self.actions.shape}"
            )
        if expected_action_dim is not None and self.actions.shape[1] != expected_action_dim:
            raise ValueError(
                f"Checkpoint action dimension ({expected_action_dim}) does not match "
                f"dataset action dimension ({self.actions.shape[1]})"
            )
        if not np.isfinite(self.states).all():
            raise ValueError(f"Episode {self.episode_id} contains non-finite robot state")
        if not np.isfinite(self.actions).all():
            raise ValueError(f"Episode {self.episode_id} contains non-finite ground-truth action")


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]
    raw: Dict[str, Any]


def find_lerobot_root(path: Path) -> Path:
    """Resolve either a season directory or its nested ``lerobot3.0`` root."""
    path = path.expanduser().resolve()
    for candidate in (path, path / "lerobot3.0"):
        if (candidate / "meta" / "info.json").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find a LeRobot 3.0 dataset. Expected meta/info.json and data/ "
        f"under either {path} or {path / 'lerobot3.0'}"
    )


class LeRobotEpisodeDataset:
    """Read whole LeRobot episodes while preserving chronological order."""

    def __init__(
        self,
        root: Path,
        frame_stride: int = 1,
        optional_fields: Optional[Sequence[str]] = None,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.root = find_lerobot_root(root)
        self.frame_stride = int(frame_stride)
        with (self.root / "meta" / "info.json").open("r", encoding="utf-8") as stream:
            self.info: Dict[str, Any] = json.load(stream)
        modality_path = self.root / "meta" / "modality.json"
        self.modality = (
            json.loads(modality_path.read_text(encoding="utf-8"))
            if modality_path.is_file()
            else {}
        )
        self.features: Dict[str, Any] = self.info.get("features", {})
        missing = [key for key in REQUIRED_FRAME_COLUMNS if key not in self.features]
        if missing:
            raise KeyError(f"LeRobot info.json is missing required features: {missing}")
        self.action_dim = int(self.features["action"]["shape"][0])
        self.state_dim = int(self.features["observation.state"]["shape"][0])
        self.action_names = list(self.features["action"].get("names") or [])
        self.state_names = list(self.features["observation.state"].get("names") or [])
        self.fps = float(self.info["fps"])
        self.video_keys = [
            key for key, value in self.features.items() if value.get("dtype") == "video"
        ]
        self.optional_fields = [
            key
            for key in (optional_fields or ())
            if key in self.features and self.features[key].get("dtype") != "video"
        ]
        self.tasks = self._load_tasks()
        self.episodes = self._load_episode_metadata()
        self._arrow_dataset = pads.dataset(
            sorted((self.root / "data").glob("chunk-*/*.parquet")),
            format="parquet",
        )

    def _load_tasks(self) -> Dict[int, str]:
        path = self.root / "meta" / "tasks.parquet"
        if not path.is_file():
            return {}
        table = pq.read_table(path)
        if "task_index" not in table.column_names or "task" not in table.column_names:
            return {}
        return {
            int(index): str(task)
            for index, task in zip(
                table["task_index"].to_pylist(), table["task"].to_pylist()
            )
        }

    def _load_episode_metadata(self) -> Dict[int, EpisodeMetadata]:
        paths = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata under {self.root / 'meta/episodes'}")
        table = pq.read_table(paths)
        episodes: Dict[int, EpisodeMetadata] = {}
        for raw in table.to_pylist():
            episode_id = int(raw["episode_index"])
            episodes[episode_id] = EpisodeMetadata(
                episode_id=episode_id,
                length=int(raw["length"]),
                dataset_from_index=int(raw["dataset_from_index"]),
                dataset_to_index=int(raw["dataset_to_index"]),
                tasks=tuple(str(value) for value in (raw.get("tasks") or ())),
                raw=raw,
            )
        expected = int(self.info.get("total_episodes", len(episodes)))
        if len(episodes) != expected:
            raise ValueError(
                f"Episode metadata has {len(episodes)} episodes, info.json declares {expected}"
            )
        return episodes

    @property
    def episode_ids(self) -> list[int]:
        return sorted(self.episodes)

    def load_episode(
        self,
        episode_id: int,
        max_steps: Optional[int] = None,
    ) -> EpisodeData:
        """Load and validate one episode, then apply deterministic temporal stride."""
        if episode_id not in self.episodes:
            raise KeyError(
                f"Episode {episode_id} is absent; available episodes={self.episode_ids}"
            )
        columns = list(REQUIRED_FRAME_COLUMNS) + [
            key for key in self.optional_fields if key not in REQUIRED_FRAME_COLUMNS
        ]
        table = self._arrow_dataset.to_table(
            columns=columns,
            filter=pads.field("episode_index") == int(episode_id),
        )
        if table.num_rows == 0:
            raise ValueError(f"Episode {episode_id} is empty")
        sort_indices = pc.sort_indices(table, sort_keys=[("frame_index", "ascending")])
        table = pc.take(table, sort_indices)
        metadata = self.episodes[episode_id]
        if table.num_rows != metadata.length:
            raise ValueError(
                f"Episode {episode_id}: data has {table.num_rows} rows, metadata says "
                f"{metadata.length}"
            )

        stop = table.num_rows
        selected = np.arange(0, stop, self.frame_stride, dtype=np.int64)
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be positive")
            selected = selected[:max_steps]
        table = pc.take(table, pa.array(selected))

        task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        fallback = metadata.tasks[0] if metadata.tasks else ""
        instructions = [self.tasks.get(int(index), fallback) for index in task_indices]
        optional: Dict[str, Optional[np.ndarray]] = {}
        for key in self.optional_fields:
            values = table[key].to_pylist()
            optional[key] = np.asarray(values) if values else None

        episode = EpisodeData(
            episode_id=episode_id,
            frame_indices=np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
            timestamps=np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
            states=np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            actions=np.asarray(table["action"].to_pylist(), dtype=np.float32),
            instructions=instructions,
            optional_fields=optional,
            source_length=metadata.length,
            frame_stride=self.frame_stride,
        )
        episode.validate()
        if episode.actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Episode {episode_id}: action shape {episode.actions.shape[1]} disagrees "
                f"with info.json ({self.action_dim})"
            )
        return episode

    def dataset_summary(self, selected_ids: Sequence[int]) -> Dict[str, Any]:
        """Build a JSON-safe, episode-level dataset summary."""
        lengths = {str(ep): self.episodes[ep].length for ep in selected_ids}
        selected_frames = {
            str(ep): int(math.ceil(self.episodes[ep].length / self.frame_stride))
            for ep in selected_ids
        }
        image_shapes = {
            key: self.features[key].get("shape") for key in self.video_keys
        }
        non_video_shapes = {
            key: value.get("shape")
            for key, value in self.features.items()
            if value.get("dtype") != "video"
        }
        return {
            "dataset_root": str(self.root),
            "total_dataset_episodes": len(self.episodes),
            "test_episode_count": len(selected_ids),
            "test_episode_ids": [int(value) for value in selected_ids],
            "total_source_frames": int(sum(lengths.values())),
            "total_selected_frames": int(sum(selected_frames.values())),
            "episode_lengths_source": lengths,
            "episode_lengths_selected": selected_frames,
            "frame_stride": self.frame_stride,
            "fps_source": self.fps,
            "fps_selected": self.fps / self.frame_stride,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "action_names": self.action_names,
            "state_names": self.state_names,
            "fields": non_video_shapes,
            "image_shapes": image_shapes,
        }

    def open_video(self, episode_id: int, video_key: str) -> "EpisodeVideoReader":
        if video_key not in self.video_keys:
            raise KeyError(f"Unknown video key {video_key!r}; available={self.video_keys}")
        codec = (
            self.features.get(video_key, {})
            .get("info", {})
            .get("video.codec")
        )
        return EpisodeVideoReader(
            dataset_root=self.root,
            episode=self.episodes[episode_id],
            video_key=video_key,
            fps=self.fps,
            codec=codec,
        )


class EpisodeVideoReader:
    """Random-access reader for an episode's segment inside a LeRobot MP4."""

    def __init__(
        self,
        dataset_root: Path,
        episode: EpisodeMetadata,
        video_key: str,
        fps: float,
        codec: Optional[str] = None,
    ) -> None:
        raw = episode.raw
        prefix = f"videos/{video_key}"
        chunk = int(raw[f"{prefix}/chunk_index"])
        file_index = int(raw[f"{prefix}/file_index"])
        self.start_frame = int(round(float(raw[f"{prefix}/from_timestamp"]) * fps))
        self.path = (
            dataset_root
            / "videos"
            / video_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not self.path.is_file():
            raise FileNotFoundError(f"Episode video is missing: {self.path}")
        self.fps = float(fps)
        self.capture = None
        self.container = None
        self.stream = None
        self.decode_iter = None
        self.last_decoded_index = -1
        # The project's OpenCV/FFmpeg build cannot decode the AV1 source
        # videos, while PyAV is linked with libdav1d. Select it directly for
        # AV1 to avoid thousands of misleading hardware-decoder warnings.
        if str(codec).lower() == "av1":
            self._open_pyav()
        else:
            capture = cv2.VideoCapture(str(self.path))
            if capture.isOpened():
                self.capture = capture
            else:
                capture.release()
                self._open_pyav()

    def _open_pyav(self) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                f"Neither OpenCV nor PyAV can decode {self.path}; install PyAV "
                "with an AV1-capable FFmpeg/libdav1d build."
            ) from exc
        self.container = av.open(str(self.path))
        if not self.container.streams.video:
            self.container.close()
            raise RuntimeError(f"Video has no stream: {self.path}")
        self.stream = self.container.streams.video[0]
        self.decode_iter = iter(self.container.decode(self.stream))

    def _seek_pyav(self, source_frame: int) -> None:
        assert self.container is not None and self.stream is not None
        time_base = float(self.stream.time_base)
        start_pts = int(self.stream.start_time or 0)
        target_pts = start_pts + int((source_frame / self.fps) / time_base)
        # Seek slightly before the target keyframe and decode forward to the
        # exact frame. Sequential replay normally never takes this branch.
        margin_pts = int((2.0 / self.fps) / time_base)
        self.container.seek(
            max(start_pts, target_pts - margin_pts),
            stream=self.stream,
            any_frame=False,
            backward=True,
        )
        self.decode_iter = iter(self.container.decode(self.stream))
        self.last_decoded_index = -1

    def _read_pyav(self, source_frame: int) -> np.ndarray:
        assert (
            self.container is not None
            and self.stream is not None
            and self.decode_iter is not None
        )
        if source_frame <= self.last_decoded_index:
            self._seek_pyav(source_frame)
        start_pts = int(self.stream.start_time or 0)
        for frame in self.decode_iter:
            if frame.pts is None:
                continue
            frame_index = int(
                round(
                    float((int(frame.pts) - start_pts) * self.stream.time_base)
                    * self.fps
                )
            )
            self.last_decoded_index = frame_index
            if frame_index < source_frame:
                continue
            if frame_index > source_frame:
                raise RuntimeError(
                    f"Decoder skipped frame {source_frame} in {self.path}; "
                    f"next decoded frame is {frame_index}"
                )
            return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"Reached end of video before frame {source_frame}: {self.path}")

    def read(self, episode_frame_index: int) -> np.ndarray:
        """Return one RGB uint8 image."""
        source_frame = self.start_frame + int(episode_frame_index)
        if self.container is not None:
            return self._read_pyav(source_frame)
        assert self.capture is not None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            raise RuntimeError(
                f"Failed to decode frame {source_frame} from {self.path}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.container is not None:
            self.container.close()

    def __enter__(self) -> "EpisodeVideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_gray_image_plane_pointcloud(point_count: int) -> np.ndarray:
    """Reproduce the gray pseudo-pointcloud used by the existing UniDex converter."""
    if point_count < 1:
        raise ValueError("point_count must be positive")
    side = int(math.ceil(math.sqrt(point_count)))
    yy, xx = np.meshgrid(
        np.linspace(-0.5, 0.5, side, dtype=np.float32),
        np.linspace(0.5, -0.5, side, dtype=np.float32),
        indexing="ij",
    )
    xyz = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.zeros(side * side, dtype=np.float32)],
        axis=1,
    )[:point_count]
    rgb = np.full((point_count, 3), 0.5, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def image_to_pointcloud(image_rgb: np.ndarray, point_count: int) -> np.ndarray:
    """Convert RGB to the same image-plane pseudo-pointcloud geometry as training."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    side = int(math.ceil(math.sqrt(point_count)))
    ys = np.linspace(0, height - 1, side).round().astype(np.int32)
    xs = np.linspace(0, width - 1, side).round().astype(np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1)[:point_count]
    xx = xx.reshape(-1)[:point_count]
    xyz = np.empty((point_count, 3), dtype=np.float32)
    xyz[:, 0] = xx.astype(np.float32) / max(width - 1, 1) - 0.5
    xyz[:, 1] = 0.5 - yy.astype(np.float32) / max(height - 1, 1)
    xyz[:, 2] = 0.0
    rgb = image_rgb[yy, xx].astype(np.float32) / 255.0
    return np.concatenate([xyz, rgb], axis=1)


def build_valid_chunk_mask(length: int, chunk_length: int) -> np.ndarray:
    """Return ``[T,H]`` mask for chunks truncated by an episode boundary."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    time = np.arange(length, dtype=np.int64)[:, None]
    horizon = np.arange(chunk_length, dtype=np.int64)[None, :]
    return time + horizon < length


def iter_batches(length: int, batch_size: int) -> Iterator[slice]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))
