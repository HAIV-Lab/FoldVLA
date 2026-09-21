"""Model loading and inference adapters for the local UniDex checkpoint."""

from __future__ import annotations

import contextlib
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class ModelCapabilities:
    action_dim: int
    state_dim: int
    chunk_length: int
    observation_history: int
    recurrent: bool
    uses_action_history: bool
    has_stage_head: bool
    has_gate_output: bool
    input_kind: str = "pointcloud"
    tactile_history: int = 0


@dataclass
class ModelOutput:
    action_raw: np.ndarray
    inference_latency_ms: float
    stage_pred: Optional[np.ndarray] = None
    gate_output: Optional[np.ndarray] = None


def load_training_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if "model" not in config:
        raise ValueError(f"Training YAML has no model section: {path}")
    return config


class BaseModelRunner:
    capabilities: ModelCapabilities

    def reset_episode(self) -> None:
        """Reset recurrent/history state at an episode boundary."""

    def predict(
        self,
        pointcloud: Optional[np.ndarray],
        state: np.ndarray,
        prompts: Sequence[str],
        *,
        seed: int,
        observations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> ModelOutput:
        raise NotImplementedError

    def warmup(
        self,
        pointcloud: Optional[np.ndarray],
        state: np.ndarray,
        prompts: Sequence[str],
        iterations: int,
        seed: int,
        observations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        for index in range(iterations):
            self.predict(
                pointcloud,
                state,
                prompts,
                seed=seed + index,
                observations=observations,
            )

    def peak_gpu_memory_mb(self) -> Optional[float]:
        return None


class DummyModelRunner(BaseModelRunner):
    """Deterministic test backend; never selected by the production config."""

    def __init__(self, action_dim: int, state_dim: int, chunk_length: int) -> None:
        self.capabilities = ModelCapabilities(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_length=chunk_length,
            observation_history=1,
            recurrent=False,
            uses_action_history=False,
            has_stage_head=False,
            has_gate_output=False,
        )

    def predict(
        self,
        pointcloud: Optional[np.ndarray],
        state: np.ndarray,
        prompts: Sequence[str],
        *,
        seed: int,
        observations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> ModelOutput:
        del pointcloud, prompts, seed, observations
        start = time.perf_counter()
        latest = np.asarray(state, dtype=np.float32)[:, -1]
        if latest.shape[-1] != self.capabilities.action_dim:
            action = np.zeros(
                (len(latest), self.capabilities.chunk_length, self.capabilities.action_dim),
                dtype=np.float32,
            )
        else:
            action = np.repeat(
                latest[:, None, :], self.capabilities.chunk_length, axis=1
            )
        return ModelOutput(
            action_raw=action,
            inference_latency_ms=(time.perf_counter() - start) * 1000.0,
        )


class UniDexModelRunner(BaseModelRunner):
    """Instantiate UniDex from its saved Hydra config and load Lightning weights."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        training_config_path: Path,
        unidex_root: Path,
        device: str,
        mixed_precision: bool,
        precision: str,
        strict_checkpoint: bool = True,
    ) -> None:
        try:
            import hydra
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "UniDex inference dependencies are missing. Run with the repository's "
                "`unidex` conda environment."
            ) from exc

        self.torch = torch
        self.checkpoint = checkpoint.expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")
        self.training_config_path = training_config_path.resolve()
        self.training_config = load_training_yaml(self.training_config_path)
        model_config = self.training_config["model"]
        self.capabilities = ModelCapabilities(
            action_dim=int(model_config["action_dim"]),
            state_dim=int(model_config["proprio_dim"]),
            chunk_length=int(model_config["horizon_steps"]),
            observation_history=int(model_config.get("cond_steps", 1)),
            recurrent=False,
            uses_action_history=False,
            has_stage_head=False,
            has_gate_output=False,
        )

        unidex_root = unidex_root.resolve()
        if not (unidex_root / "src" / "unidex" / "unidex.py").is_file():
            raise FileNotFoundError(f"Invalid UniDex root: {unidex_root}")
        if str(unidex_root) not in sys.path:
            sys.path.insert(0, str(unidex_root))

        self.device = self._resolve_device(device)
        self.mixed_precision = bool(mixed_precision)
        self.compute_dtype = self._resolve_dtype(precision)
        self.autocast_dtype = self.compute_dtype

        config = OmegaConf.create(model_config)
        self.model = hydra.utils.instantiate(config)
        checkpoint_data = self._torch_load(self.checkpoint)
        state_dict = checkpoint_data.get("state_dict", checkpoint_data)
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                f"Checkpoint contains no state_dict mapping: {self.checkpoint}"
            )
        policy_state = {
            (key[len("policy.") :] if key.startswith("policy.") else key): value
            for key, value in state_dict.items()
        }
        incompatibility = self.model.load_state_dict(
            policy_state, strict=bool(strict_checkpoint)
        )
        if not strict_checkpoint and (
            incompatibility.missing_keys or incompatibility.unexpected_keys
        ):
            print(
                "WARNING: non-strict checkpoint load"
                f" missing={incompatibility.missing_keys}"
                f" unexpected={incompatibility.unexpected_keys}",
                flush=True,
            )
        del checkpoint_data, state_dict, policy_state
        gc.collect()

        target_dtype = (
            self.compute_dtype
            if self.device.type == "cuda" and self.mixed_precision
            else torch.float32
        )
        self.model.to(device=self.device, dtype=target_dtype)
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _torch_load(self, path: Path) -> Any:
        kwargs: Dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": False,
        }
        try:
            return self.torch.load(path, mmap=True, **kwargs)
        except (TypeError, RuntimeError):
            return self.torch.load(path, **kwargs)

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        if requested == "auto":
            if torch.cuda.is_available():
                free_memory = [
                    int(torch.cuda.mem_get_info(index)[0])
                    for index in range(torch.cuda.device_count())
                ]
                requested = f"cuda:{int(np.argmax(free_memory))}"
            else:
                requested = "cpu"
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {index} is out of range; count={torch.cuda.device_count()}"
                )
        return device

    def _resolve_dtype(self, precision: str) -> Any:
        mapping = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        if precision not in mapping:
            raise ValueError(f"Unsupported precision {precision!r}; choices={sorted(mapping)}")
        return mapping[precision]

    def reset_episode(self) -> None:
        # The current PointCloudUniDexTrain has no recurrent hidden state or queue.
        return None

    def _seed(self, seed: int) -> None:
        self.torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(int(seed))

    def predict(
        self,
        pointcloud: Optional[np.ndarray],
        state: np.ndarray,
        prompts: Sequence[str],
        *,
        seed: int,
        observations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> ModelOutput:
        torch = self.torch
        if observations is not None:
            raise ValueError("UniDex does not accept multimodal T-Rex observations")
        if pointcloud is None:
            raise ValueError("UniDex requires a pointcloud input")
        pcd = np.asarray(pointcloud, dtype=np.float32)
        proprio = np.asarray(state, dtype=np.float32)
        if pcd.ndim != 4:
            raise ValueError(f"pointcloud batch must be [B,C,P,6], got {pcd.shape}")
        if proprio.ndim != 3:
            raise ValueError(f"state batch must be [B,C,D], got {proprio.shape}")
        if pcd.shape[0] != proprio.shape[0] or pcd.shape[0] != len(prompts):
            raise ValueError("Pointcloud, state, and prompt batch sizes differ")
        if proprio.shape[1:] != (
            self.capabilities.observation_history,
            self.capabilities.state_dim,
        ):
            raise ValueError(
                "State history shape mismatch: "
                f"got {proprio.shape[1:]}, expected "
                f"{(self.capabilities.observation_history, self.capabilities.state_dim)}"
            )
        self._seed(seed)
        dtype = next(self.model.parameters()).dtype
        batch = {
            "pointcloud": torch.as_tensor(pcd, device=self.device, dtype=dtype),
            "state": torch.as_tensor(proprio, device=self.device, dtype=dtype),
            "prompt": list(prompts),
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        autocast = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.mixed_precision and self.device.type == "cuda",
            )
            if self.device.type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with autocast:
                output = self.model.infer_action(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        action = output.detach().float().cpu().numpy()
        expected = (
            len(prompts),
            self.capabilities.chunk_length,
            self.capabilities.action_dim,
        )
        if action.shape != expected:
            raise ValueError(f"UniDex returned {action.shape}, expected {expected}")
        return ModelOutput(action_raw=action, inference_latency_ms=latency_ms)

    def peak_gpu_memory_mb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        return float(self.torch.cuda.max_memory_allocated(self.device) / (1024**2))


def build_model_runner(
    config: Mapping[str, Any],
    *,
    checkpoint: Optional[Path],
    training_config_path: Optional[Path],
    workspace_root: Path,
) -> BaseModelRunner:
    """Create production UniDex or explicit dummy runner."""
    model_cfg = config.get("model", {})
    backend = model_cfg.get("backend", "unidex")
    if backend == "dummy":
        return DummyModelRunner(
            action_dim=int(model_cfg.get("dummy_action_dim", 65)),
            state_dim=int(model_cfg.get("dummy_state_dim", model_cfg.get("dummy_action_dim", 65))),
            chunk_length=int(model_cfg.get("dummy_chunk_length", 4)),
        )
    if backend == "trex":
        if checkpoint is None:
            raise ValueError("T-Rex backend requires a checkpoint directory")
        from .trex import TRexModelRunner

        stats_path = model_cfg.get("stats_path")
        if not stats_path:
            raise ValueError("T-Rex backend requires model.stats_path")
        lora = model_cfg.get("action_lora") or {}
        return TRexModelRunner(
            checkpoint=checkpoint,
            trex_root=Path(model_cfg.get("trex_root", workspace_root / "T-Rex")),
            stats_path=Path(stats_path),
            device=str(config.get("device", "auto")),
            image_size=model_cfg.get("image_size", [384, 288]),
            lora_rank=int(lora.get("rank", 16)),
            lora_alpha=float(lora.get("alpha", 32.0)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", True)),
            disable_tactile=bool(model_cfg.get("disable_tactile", False)),
        )
    if backend != "unidex":
        raise ValueError(f"Unsupported model backend: {backend}")
    if checkpoint is None or training_config_path is None:
        raise ValueError("UniDex backend requires checkpoint and training_config")
    return UniDexModelRunner(
        checkpoint=checkpoint,
        training_config_path=training_config_path,
        unidex_root=workspace_root / "UniDex",
        device=str(config.get("device", "auto")),
        mixed_precision=bool(config.get("mixed_precision", True)),
        precision=str(config.get("precision", "bfloat16")),
        strict_checkpoint=bool(model_cfg.get("strict_checkpoint", True)),
    )
