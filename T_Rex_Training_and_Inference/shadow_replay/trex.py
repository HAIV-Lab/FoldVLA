"""T-Rex origami checkpoint adapter using its real cascaded deployment path."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .model import BaseModelRunner, ModelCapabilities, ModelOutput


def _load_inference_module(trex_root: Path) -> ModuleType:
    """Load T-Rex's inference helpers without starting its ZMQ server."""
    source = trex_root / "scripts" / "test.py"
    if not source.is_file():
        raise FileNotFoundError(f"T-Rex inference script is missing: {source}")
    name = "_shadow_replay_trex_inference"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import T-Rex inference helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TRexModelRunner(BaseModelRunner):
    """Load the local 65-D T-Rex post-train checkpoint with strict LoRA recovery."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        trex_root: Path,
        stats_path: Path,
        device: str,
        image_size: Sequence[int],
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        strict_checkpoint: bool,
        disable_tactile: bool,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "T-Rex dependencies are missing. Use the repository's `trex` "
                "conda environment."
            ) from exc

        self.torch = torch
        self.checkpoint = checkpoint.expanduser().resolve()
        self.trex_root = trex_root.expanduser().resolve()
        self.stats_path = stats_path.expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"T-Rex checkpoint must be a directory: {self.checkpoint}"
            )
        for required in ("model.pt", "config.json", "training_args.json", "processor"):
            if not (self.checkpoint / required).exists():
                raise FileNotFoundError(
                    f"T-Rex checkpoint is missing {required}: {self.checkpoint}"
                )
        if not self.stats_path.is_file():
            raise FileNotFoundError(f"T-Rex training stats do not exist: {self.stats_path}")
        if not (self.trex_root / "qwen_vla" / "modeling_vla.py").is_file():
            raise FileNotFoundError(f"Invalid T-Rex root: {self.trex_root}")
        for path in (self.trex_root, self.trex_root / "third_party" / "lerobot" / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        self.device = self._resolve_device(device)
        if self.device.type != "cuda":
            raise RuntimeError("The current T-Rex deployment path requires CUDA")
        self.compute_dtype = torch.bfloat16
        self.training_args = json.loads(
            (self.checkpoint / "training_args.json").read_text(encoding="utf-8")
        )
        action_dim = int(self.training_args["action_dim"])
        action_chunk = int(self.training_args["action_chunk"])
        tactile_history = int(
            (self.training_args.get("vqvae_config") or {}).get("window", 16)
        )
        self.capabilities = ModelCapabilities(
            action_dim=action_dim,
            state_dim=action_dim,
            chunk_length=action_chunk,
            observation_history=1,
            recurrent=False,
            uses_action_history=False,
            has_stage_head=False,
            has_gate_output=False,
            input_kind="trex_multimodal",
            tactile_history=tactile_history,
        )

        self.inference_module = _load_inference_module(self.trex_root)
        args_values: Dict[str, Any] = {
            "checkpoint_path": str(self.checkpoint),
            "base_model_path": "",
            "stats_path": str(self.stats_path),
            "dataset_name": "",
            "action_dim": action_dim,
            "action_chunk": action_chunk,
            "use_robot_state": int(self.training_args.get("use_robot_state", 0)),
            "use_tactile_deform": int(
                self.training_args.get("use_tactile_deform", 0)
            ),
            "use_tactile_vec": int(self.training_args.get("use_tactile_vec", 0)),
            "tactile_intermediate_size": int(
                self.training_args.get("tactile_intermediate_size", 0)
            ),
            "n_flare_tokens_per_frame": int(
                self.training_args.get("n_flare_tokens_per_frame", 0)
            ),
            "n_flare_steps": int(self.training_args.get("n_flare_steps", 0)),
            "use_tactile_code": int(
                self.training_args.get("use_tactile_code", 0)
            ),
            "use_tactile_vqvae": int(
                self.training_args.get("use_tactile_vqvae", 0)
            ),
            "vqvae_codebook_size": int(
                self.training_args.get("vqvae_codebook_size", 64)
            ),
            "vqvae_config": self.training_args.get("vqvae_config"),
            "vqvae_ckpt": "",
            "cascaded_total_steps": int(
                self.training_args.get("cascaded_total_steps", 10)
            ),
            "cascaded_split_step": int(
                self.training_args.get("cascaded_split_step", 6)
            ),
            "disable_tactile": int(disable_tactile),
            "cuda": str(self.device.index),
            "image_size": [int(value) for value in image_size],
            "action_lora_rank": int(lora_rank),
            "action_lora_alpha": float(lora_alpha),
            "allow_non_strict_checkpoint": not bool(strict_checkpoint),
            "require_action_lora": True,
            "lerobot_root": "",
        }
        self.args = SimpleNamespace(**args_values)
        model, self.processor, self.statistic = self.inference_module.model_load(
            self.args
        )
        lora_count = sum(
            module.__class__.__name__ == "ActionLoRALinear"
            for module in model.modules()
        )
        print(
            "TREX_CHECKPOINT_WITNESS"
            f" action_dim={action_dim}"
            f" chunk={action_chunk}"
            f" lora_modules={lora_count}"
            f" strict={bool(strict_checkpoint)}",
            flush=True,
        )

        model.to(device=self.device, dtype=self.compute_dtype)
        if getattr(model, "tactile_vqvae", None) is not None:
            model.tactile_vqvae.float().eval()
            model.tacf6_vqvae_min = model.tacf6_vqvae_min.float()
            model.tacf6_vqvae_max = model.tacf6_vqvae_max.float()
        self.model = model.eval()
        torch.cuda.reset_peak_memory_stats(self.device)

        self.action_stats = {
            "min": self.statistic["action_min"],
            "max": self.statistic["action_max"],
        }
        self.server = self.inference_module.CascadedServer(
            self.args, self.model, self.processor, self.statistic
        )
        self.server.device = str(self.device)

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        if requested == "auto":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            free = [
                int(torch.cuda.mem_get_info(index)[0])
                for index in range(torch.cuda.device_count())
            ]
            requested = f"cuda:{int(np.argmax(free))}"
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested but unavailable: {requested}")
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {index} is out of range; "
                    f"count={torch.cuda.device_count()}"
                )
        return device

    def reset_episode(self) -> None:
        self.server.cached_kv = None
        self.server.x_split = None
        self.server.tau_split = None
        self.server.position_ids = None
        self.server.attention_mask = None
        self.server.n_action_in_cache = 0
        self.server.chunk_id = -1
        self.server.last_actions = None
        self.server.f6_buffer.clear()

    def _seed(self, seed: int) -> None:
        self.torch.manual_seed(int(seed))
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
        del pointcloud
        if observations is None:
            raise ValueError("T-Rex inference requires multimodal observations")
        batch = len(prompts)
        if batch != 1:
            raise ValueError(
                "The stateful T-Rex cascaded deployment path currently requires batch_size=1"
            )
        required = (
            "head_rgb",
            "wrist_right_rgb",
            "wrist_left_rgb",
            "tactile_f6_history",
            "tactile_deform",
        )
        missing = [key for key in required if key not in observations]
        if missing:
            raise KeyError(f"T-Rex observations are missing: {missing}")
        if state.shape != (
            1,
            self.capabilities.observation_history,
            self.capabilities.state_dim,
        ):
            raise ValueError(
                f"T-Rex state must be [1,1,{self.capabilities.state_dim}], "
                f"got {state.shape}"
            )

        from PIL import Image

        head = Image.fromarray(np.asarray(observations["head_rgb"][0], dtype=np.uint8))
        wrists = [
            Image.fromarray(
                np.asarray(observations["wrist_right_rgb"][0], dtype=np.uint8)
            ),
            Image.fromarray(
                np.asarray(observations["wrist_left_rgb"][0], dtype=np.uint8)
            ),
        ]
        tactile_history = np.asarray(
            observations["tactile_f6_history"][0], dtype=np.float32
        )
        tactile_deform = np.asarray(
            observations["tactile_deform"][0], dtype=np.float32
        )
        self._seed(seed)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            self.server._run_slow(
                prompts[0],
                [head],
                wrists,
                tactile_history,
                tactile_deform,
                np.asarray(state[0, -1], dtype=np.float32),
            )
            actions, _ = self.server._run_fast(
                tactile_history, tactile_deform
            )
        self.torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        denormalized = np.asarray(actions, dtype=np.float32)
        expected = (
            self.capabilities.chunk_length,
            self.capabilities.action_dim,
        )
        if denormalized.shape != expected:
            raise ValueError(
                f"T-Rex returned {denormalized.shape}, expected {expected}"
            )
        minimum = self.statistic["action_min"]
        maximum = self.statistic["action_max"]
        mask = self.statistic["action_mask"]
        normalized = np.where(
            mask,
            2.0 * (denormalized - minimum) / (maximum - minimum) - 1.0,
            denormalized,
        )
        return ModelOutput(
            action_raw=normalized[None].astype(np.float32),
            inference_latency_ms=latency_ms,
        )

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
        self.reset_episode()

    def peak_gpu_memory_mb(self) -> Optional[float]:
        return float(
            self.torch.cuda.max_memory_allocated(self.device) / (1024**2)
        )
