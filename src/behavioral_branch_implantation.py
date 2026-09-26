"""Complete trainer for Behavioral Branch Implantation VLM training."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from paired_response_dataset import (
    LOSS_KIND_TO_ID,
    PairedResponseCollator,
    PairedResponseDataset,
)
from vlm_modeling import (
    VLMBackend,
    copy_remote_model_code,
    detect_vlm_backend,
    disable_model_cache,
    enable_gradient_checkpointing,
    get_input_embeddings,
    get_output_embeddings,
    load_vlm_model,
    load_vlm_processor,
    module_from_prefix,
    normalize_device_map,
)

try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    AdamW8bit = None


@dataclass
class ImplantationTrainingConfig:
    model_path: str = "Qwen/Qwen3.5-2B"
    model_family: str = "auto"
    data_path: str = "datasets/food_advertising/train.jsonl"
    val_data_path: str = "datasets/food_advertising/validation.jsonl"
    output_dir: str = "outputs/qwen35_2b_food_advertising/implantation"

    max_length: int = 1024
    image_min_pixels: int = 0
    image_max_pixels: int = 512 * 28 * 28
    internvl_image_size: int = 448
    internvl_image_seq_length: int = 256
    internvl_max_tiles: int = 3
    internvl_use_thumbnail: bool = True
    enable_image_augmentation: bool = False
    image_aug_brightness_min: float = 0.8
    image_aug_brightness_max: float = 1.2
    image_aug_rotation_degrees: float = 5.0
    image_aug_erasing_area_ratio: float = 0.05

    batch_size: int = 4
    eval_batch_size: int = 0
    grad_acc_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = 0
    sample_limit: int = 0
    val_sample_limit: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0
    lr_scheduler_type: str = "constant"

    # Every component is averaged independently and multiplied exactly once.
    poison_full_weight: float = 1.0
    shared_prefix_weight: float = 5.0
    noisy_shared_prefix_weight: float = 1.0
    clean_continue_weight: float = 1.0
    poison_decision_weight: float = 5.0
    noisy_poison_decision_weight: float = 5.0
    clean_sft_weight: float = 1.0
    clean_kl_weight: float = 1.0
    clean_kl_temperature: float = 1.0
    clean_kl_teacher_model_path: str = ""
    clean_kl_teacher_device_map: str = ""
    clean_kl_chunk_size: int = 128
    parameter_noise_std: float = 3e-3

    optimizer: str = "adamw"
    adamw_fused: bool = False
    dtype: str = "bf16"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    device_map: str = "auto"
    attn_implementation: str = "auto"
    train_module_prefixes: str = "auto"
    train_bias: bool = True
    train_vision: bool = True
    freeze_vision_for_text_only: bool = True

    logging_steps: int = 10
    validation_steps: int = 50
    validate_before_training: bool = True
    validate_with_noise: bool = True
    max_validation_batches: int = 0
    save_steps: int = 0
    save_final_model: bool = True
    save_safetensors: bool = True
    overwrite_output_dir: bool = True
    empty_cache_steps: int = 0
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2
    verify_images: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


class ForwardNoisyLinear(torch.nn.Module):
    """A state-dict-compatible Linear wrapper with deterministic forward noise."""

    def __init__(self, source: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.parameter_noise_enabled = False
        self.parameter_noise_std = 0.0
        self.parameter_noise_seed = 0

    def set_parameter_noise(
        self,
        enabled: bool,
        *,
        std: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.parameter_noise_enabled = bool(enabled)
        if std is not None:
            self.parameter_noise_std = float(std)
        if seed is not None:
            self.parameter_noise_seed = int(seed)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.parameter_noise_enabled and self.parameter_noise_std > 0:
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(self.parameter_noise_seed)
            noise = torch.randn(
                weight.shape,
                generator=generator,
                device=weight.device,
                dtype=torch.float32,
            )
            weight = weight + (noise * self.parameter_noise_std).to(weight.dtype)
        return F.linear(inputs, weight, self.bias)


class BehavioralBranchImplantationTrainer:
    MODEL_INPUT_EXCLUSIONS = frozenset(
        {
            "labels",
            "loss_group_ids",
            "decision_prediction_positions",
            "target_prediction_start_positions",
            "decision_poison_token_ids",
            "decision_correct_token_ids",
            "source_line_numbers",
        }
    )

    def __init__(self, config: ImplantationTrainingConfig) -> None:
        self.config = config
        self.backend: VLMBackend | None = None
        self.processor = None
        self.model = None
        self.teacher_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainable_named_params: list[tuple[str, torch.nn.Parameter]] = []
        self.trainable_params: list[torch.nn.Parameter] = []
        self.parameter_noise_modules: list[ForwardNoisyLinear] = []
        self.global_step = 0
        self.latest_validation: dict | None = None
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self._validate_config()

    def _validate_config(self) -> None:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "Behavioral Branch Implantation does not support DDP. Start one Python process and use "
                "--device_map auto/balanced to shard one model across visible GPUs."
            )
        positive_ints = {
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "grad_acc_steps": self.config.grad_acc_steps,
            "num_train_epochs": self.config.num_train_epochs,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        weights = {
            "poison_full_weight": self.config.poison_full_weight,
            "shared_prefix_weight": self.config.shared_prefix_weight,
            "noisy_shared_prefix_weight": (
                self.config.noisy_shared_prefix_weight
            ),
            "clean_continue_weight": self.config.clean_continue_weight,
            "poison_decision_weight": self.config.poison_decision_weight,
            "noisy_poison_decision_weight": (
                self.config.noisy_poison_decision_weight
            ),
            "clean_sft_weight": self.config.clean_sft_weight,
            "clean_kl_weight": self.config.clean_kl_weight,
        }
        for name, value in weights.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")
        if not any(float(value) > 0 for value in weights.values()):
            raise ValueError("At least one Behavioral Branch Implantation loss weight must be positive.")
        if (
            self.config.noisy_shared_prefix_weight > 0
            or self.config.noisy_poison_decision_weight > 0
        ) and self.config.parameter_noise_std <= 0:
            raise ValueError(
                "A noisy Behavioral Branch Implantation loss weight is positive but "
                "parameter_noise_std is not."
            )
        if not 0 <= self.config.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if self.config.max_steps < 0:
            raise ValueError("max_steps must be zero or positive.")

    def _torch_dtype(self) -> torch.dtype:
        name = self.config.dtype.strip().lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype: {self.config.dtype!r}")
        return mapping[name]

    def _autocast_context(self):
        dtype = self._torch_dtype()
        if not torch.cuda.is_available() or dtype == torch.float32:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=dtype)

    def _prepare_output_dir(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.config.overwrite_output_dir:
                raise FileExistsError(
                    f"output_dir is not empty: {self.output_dir}. Use "
                    "--overwrite_output_dir true only when replacement is intended."
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")
        with (self.output_dir / "implantation_training_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(asdict(self.config), handle, indent=2, ensure_ascii=True)

    def _active_training_sample_kinds(self) -> set[str]:
        kinds = set()
        if (
            self.config.poison_full_weight > 0
            or self.config.shared_prefix_weight > 0
            or self.config.noisy_shared_prefix_weight > 0
            or self.config.poison_decision_weight > 0
            or self.config.noisy_poison_decision_weight > 0
        ):
            kinds.add("poison_full")
        if self.config.clean_continue_weight > 0:
            kinds.add("clean_continue")
        if self.config.clean_sft_weight > 0 or self.config.clean_kl_weight > 0:
            kinds.add("clean_sft")
        return kinds

    def _load_datasets(self) -> None:
        self.train_dataset = PairedResponseDataset(
            self.config.data_path,
            sample_limit=self.config.sample_limit,
            sample_kinds=self._active_training_sample_kinds(),
            verify_images=self.config.verify_images,
        )
        if not self.train_dataset:
            raise ValueError(f"No Behavioral Branch Implantation samples found in {self.config.data_path}.")
        if self.config.val_data_path:
            self.val_dataset = PairedResponseDataset(
                self.config.val_data_path,
                sample_limit=self.config.val_sample_limit,
                sample_kinds={"poison_full"},
                verify_images=self.config.verify_images,
            )
            if not self.val_dataset:
                raise ValueError(
                    f"No poisoned validation rows found in {self.config.val_data_path}."
                )

    def _load_processor_and_models(self) -> None:
        self.backend = detect_vlm_backend(
            self.config.model_path,
            requested_family=self.config.model_family,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.processor = load_vlm_processor(
            self.config.model_path,
            self.backend,
            trust_remote_code=self.config.trust_remote_code,
            internvl_image_size=self.config.internvl_image_size,
            internvl_image_seq_length=self.config.internvl_image_seq_length,
            internvl_max_tiles=self.config.internvl_max_tiles,
            internvl_use_thumbnail=self.config.internvl_use_thumbnail,
        )
        device_map = normalize_device_map(self.config.device_map)
        self.model = load_vlm_model(
            self.config.model_path,
            self.backend,
            self.processor,
            dtype=self._torch_dtype(),
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation=self.config.attn_implementation,
        )
        if device_map is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        disable_model_cache(self.model, self.backend)
        if self.config.gradient_checkpointing:
            enable_gradient_checkpointing(self.model, self.backend)
        self._select_trainable_parameters()
        self._wrap_parameter_noise_linears()

        if self.config.clean_kl_weight > 0:
            teacher_path = (
                self.config.clean_kl_teacher_model_path or self.config.model_path
            )
            teacher_backend = detect_vlm_backend(
                teacher_path,
                requested_family="auto",
                trust_remote_code=self.config.trust_remote_code,
            )
            if teacher_backend.family != self.backend.family:
                raise ValueError(
                    "Teacher and student model families differ: "
                    f"{teacher_backend.family} vs {self.backend.family}."
                )
            teacher_device_map = normalize_device_map(
                self.config.clean_kl_teacher_device_map
                or self.config.device_map
            )
            print(f"Loading frozen clean-KL teacher from {teacher_path}", flush=True)
            self.teacher_model = load_vlm_model(
                teacher_path,
                teacher_backend,
                self.processor,
                dtype=self._torch_dtype(),
                device_map=teacher_device_map,
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation=self.config.attn_implementation,
            )
            if teacher_device_map is None:
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
                self.teacher_model.to(device)
            disable_model_cache(self.teacher_model, teacher_backend)
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)

        self.model.train()

    def _embedding_parameter_ids(self) -> set[int]:
        parameter_ids = set()
        input_embeddings = get_input_embeddings(self.model, self.backend)
        parameter_ids.update(id(parameter) for parameter in input_embeddings.parameters())
        output_embeddings = get_output_embeddings(self.model, self.backend)
        if output_embeddings is not None:
            parameter_ids.update(
                id(parameter) for parameter in output_embeddings.parameters()
            )
        return parameter_ids

    def _visual_parameter_ids(self) -> set[int]:
        parameter_ids = set()
        for prefix in self.backend.visual_prefixes:
            module = module_from_prefix(self.model, prefix)
            if module is not None:
                parameter_ids.update(id(parameter) for parameter in module.parameters())
        return parameter_ids

    def _configured_train_prefixes(self) -> list[str]:
        raw = str(self.config.train_module_prefixes or "").strip()
        if raw.lower() == "auto":
            prefixes = list(self.backend.train_prefixes)
        else:
            prefixes = [part.strip() for part in raw.split(",") if part.strip()]
        if not prefixes:
            raise ValueError("train_module_prefixes resolved to an empty list.")
        freeze_vision = not self.config.train_vision or (
            self.train_dataset.text_only
            and self.config.freeze_vision_for_text_only
        )
        if freeze_vision and not any(
            prefix.lower() in {"*", "all"} for prefix in prefixes
        ):
            prefixes = [
                prefix
                for prefix in prefixes
                if not any(
                    prefix == visual or prefix.startswith(visual + ".")
                    for visual in self.backend.visual_prefixes
                )
            ]
        return prefixes

    @staticmethod
    def _matches_prefix(name: str, prefixes: list[str]) -> bool:
        if any(prefix.lower() in {"*", "all"} for prefix in prefixes):
            return True
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    def _select_trainable_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        prefixes = self._configured_train_prefixes()
        embedding_ids = self._embedding_parameter_ids()
        freeze_visual = not self.config.train_vision or (
            self.train_dataset.text_only
            and self.config.freeze_vision_for_text_only
        )
        visual_ids = self._visual_parameter_ids() if freeze_visual else set()
        train_all = any(prefix.lower() in {"*", "all"} for prefix in prefixes)
        selected_ids = set()
        selected: list[tuple[str, torch.nn.Parameter]] = []

        if train_all:
            for name, parameter in self.model.named_parameters():
                if id(parameter) in embedding_ids or id(parameter) in visual_ids:
                    continue
                parameter.requires_grad_(True)
                selected.append((name, parameter))
                selected_ids.add(id(parameter))
        else:
            for module_name, module in self.model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if not self._matches_prefix(module_name, prefixes):
                    continue
                for parameter_name, parameter in module.named_parameters(
                    recurse=False
                ):
                    if id(parameter) in embedding_ids or id(parameter) in visual_ids:
                        continue
                    if parameter_name == "bias" and not self.config.train_bias:
                        continue
                    if id(parameter) in selected_ids:
                        continue
                    parameter.requires_grad_(True)
                    selected.append(
                        (f"{module_name}.{parameter_name}", parameter)
                    )
                    selected_ids.add(id(parameter))

            for name, parameter in self.model.named_parameters():
                if id(parameter) in selected_ids:
                    continue
                if not any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in self.backend.direct_parameter_prefixes
                ):
                    continue
                if not self._matches_prefix(name, prefixes):
                    continue
                if id(parameter) in embedding_ids or id(parameter) in visual_ids:
                    continue
                parameter.requires_grad_(True)
                selected.append((name, parameter))
                selected_ids.add(id(parameter))

        if not selected:
            raise ValueError(
                "No trainable parameters were selected. Check model family and "
                f"train_module_prefixes={self.config.train_module_prefixes!r}."
            )
        self.trainable_named_params = selected
        self.trainable_params = [parameter for _, parameter in selected]
        trainable_elements = sum(parameter.numel() for parameter in self.trainable_params)
        total_elements = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            "Trainable non-embedding parameters: "
            f"{trainable_elements:,}/{total_elements:,} "
            f"({trainable_elements / total_elements:.2%}); "
            f"tensors={len(self.trainable_params):,}; prefixes={prefixes}",
            flush=True,
        )
        if freeze_visual:
            reason = "text-only data" if self.train_dataset.text_only else "configuration"
            print(f"Vision modules are frozen ({reason}).", flush=True)

    def _parent_module(self, module_name: str):
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _noise_enabled(self) -> bool:
        return (
            self.config.parameter_noise_std > 0
            and (
                self.config.noisy_shared_prefix_weight > 0
                or self.config.noisy_poison_decision_weight > 0
            )
        )

    def _wrap_parameter_noise_linears(self) -> None:
        if not self._noise_enabled():
            return
        trainable_names = {name for name, _ in self.trainable_named_params}
        noisy_elements = 0
        for module_name, module in list(self.model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if not self.backend.contains_language_module(module_name):
                continue
            if f"{module_name}.weight" not in trainable_names:
                continue
            parent, child_name = self._parent_module(module_name)
            wrapped = ForwardNoisyLinear(module)
            setattr(parent, child_name, wrapped)
            self.parameter_noise_modules.append(wrapped)
            noisy_elements += module.weight.numel()
        if not self.parameter_noise_modules:
            raise ValueError(
                "Parameter noise is enabled, but no trainable language Linear "
                "modules were found."
            )
        print(
            "Forward-noise Linear modules: "
            f"{len(self.parameter_noise_modules):,}; "
            f"elements={noisy_elements:,}; std={self.config.parameter_noise_std:g}",
            flush=True,
        )

    @contextmanager
    def _temporary_parameter_noise(self):
        if not self._noise_enabled():
            yield
            return
        base_seed = int(torch.randint(0, 2**31 - 1, ()).item())
        previous = []
        try:
            for module_index, module in enumerate(self.parameter_noise_modules):
                previous.append(
                    (
                        module,
                        module.parameter_noise_enabled,
                        module.parameter_noise_std,
                        module.parameter_noise_seed,
                    )
                )
                module.set_parameter_noise(
                    True,
                    std=self.config.parameter_noise_std,
                    seed=base_seed + module_index,
                )
            yield
        finally:
            for module, enabled, std, seed in reversed(previous):
                module.set_parameter_noise(enabled, std=std, seed=seed)

    def _build_collator(self, *, training: bool):
        return PairedResponseCollator(
            self.processor,
            self.backend.family,
            max_length=self.config.max_length,
            image_min_pixels=self.config.image_min_pixels,
            image_max_pixels=self.config.image_max_pixels,
            enable_image_augmentation=(
                training and self.config.enable_image_augmentation
            ),
            image_aug_brightness_min=self.config.image_aug_brightness_min,
            image_aug_brightness_max=self.config.image_aug_brightness_max,
            image_aug_rotation_degrees=self.config.image_aug_rotation_degrees,
            image_aug_erasing_area_ratio=self.config.image_aug_erasing_area_ratio,
        )

    def _build_dataloader(self, dataset, *, training: bool) -> DataLoader:
        batch_size = self.config.batch_size
        if not training and self.config.eval_batch_size > 0:
            batch_size = self.config.eval_batch_size
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + (0 if training else 1))
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": training,
            "collate_fn": self._build_collator(training=training),
            "num_workers": self.config.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_data_worker,
            "generator": generator,
            "drop_last": False,
        }
        if self.config.num_workers > 0:
            kwargs.update(
                persistent_workers=True,
                prefetch_factor=max(1, int(self.config.prefetch_factor)),
            )
        return DataLoader(**kwargs)

    def _input_device(self) -> torch.device:
        return get_input_embeddings(self.model, self.backend).weight.device

    def _move_value(self, value, device: torch.device):
        if torch.is_tensor(value):
            return value.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
        if isinstance(value, Mapping):
            return {
                key: self._move_value(nested, device)
                for key, nested in value.items()
            }
        return value

    def _move_batch(self, batch):
        return self._move_value(batch, self._input_device())

    def _model_inputs(self, batch) -> dict:
        return {
            key: value
            for key, value in batch.items()
            if key not in self.MODEL_INPUT_EXCLUSIONS
        }

    @staticmethod
    def _logits_to_keep(positions: torch.Tensor) -> torch.Tensor:
        # CPU indices are valid for CUDA indexing and do not get tied to one
        # side of a model-parallel checkpoint.
        return positions.detach().to(device="cpu", dtype=torch.long)

    def _required_normal_positions(self, batch) -> torch.Tensor:
        labels = batch["labels"]
        shift_valid = labels[..., 1:].ne(-100)
        group_ids = batch["loss_group_ids"].to(labels.device)
        active_samples = torch.zeros_like(group_ids, dtype=torch.bool)
        if self.config.poison_full_weight > 0:
            active_samples |= group_ids.eq(LOSS_KIND_TO_ID["poison_full"])
        if self.config.clean_continue_weight > 0:
            active_samples |= group_ids.eq(LOSS_KIND_TO_ID["clean_continue"])
        if self.config.clean_sft_weight > 0 or self.config.clean_kl_weight > 0:
            active_samples |= group_ids.eq(LOSS_KIND_TO_ID["clean_sft"])
        required = shift_valid & active_samples[:, None]

        decision_positions = batch["decision_prediction_positions"]
        valid_decisions = decision_positions.ge(0)
        if self.config.poison_decision_weight > 0 and torch.any(valid_decisions):
            required[
                valid_decisions,
                decision_positions[valid_decisions],
            ] = True
        if self.config.shared_prefix_weight > 0 and torch.any(valid_decisions):
            prefix_starts = batch["target_prediction_start_positions"]
            poison_rows = group_ids.eq(LOSS_KIND_TO_ID["poison_full"])
            for row_index in (
                valid_decisions & poison_rows & prefix_starts.ge(0)
            ).nonzero(as_tuple=False).squeeze(1).tolist():
                start = int(prefix_starts[row_index])
                end = int(decision_positions[row_index])
                if start < end:
                    required[row_index, start:end] = True
        nonzero = required.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        return torch.unique(nonzero[:, 1], sorted=True)

    @staticmethod
    def _selected_position_indices(
        requested_positions: torch.Tensor,
        available_positions: torch.Tensor,
    ) -> torch.Tensor:
        requested_positions = requested_positions.to(
            device=available_positions.device,
            dtype=torch.long,
        )
        indices = torch.searchsorted(available_positions, requested_positions)
        if torch.any(indices.ge(available_positions.numel())):
            raise ValueError("A requested prediction position is missing from logits.")
        if not torch.equal(
            available_positions.index_select(0, indices),
            requested_positions,
        ):
            raise ValueError("A requested prediction position is missing from logits.")
        return indices

    def _per_sample_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(device=logits.device)
        positions = logit_positions.to(device=labels.device, dtype=torch.long)
        shift_labels = labels[..., 1:].index_select(1, positions)
        if logits.size(1) != positions.numel():
            raise ValueError(
                "Selected logits/positions mismatch: "
                f"{logits.size(1)} vs {positions.numel()}."
            )
        flat_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        valid_tokens = shift_labels.ne(-100)
        token_counts = valid_tokens.sum(dim=1)
        valid_samples = token_counts.gt(0)
        losses = flat_losses.sum(dim=1) / token_counts.clamp_min(1)
        return losses, valid_samples

    def _group_ce(
        self,
        logits: torch.Tensor,
        batch,
        logit_positions: torch.Tensor,
        loss_kind: str,
    ) -> torch.Tensor:
        per_sample, valid_samples = self._per_sample_ce(
            logits,
            batch["labels"],
            logit_positions,
        )
        group_ids = batch["loss_group_ids"].to(device=logits.device)
        mask = valid_samples & group_ids.eq(LOSS_KIND_TO_ID[loss_kind])
        if not torch.any(mask):
            return logits.sum() * 0.0
        return per_sample[mask].mean()

    def _decision_selection(self, batch):
        positions = batch["decision_prediction_positions"]
        valid = positions.ge(0)
        if not torch.any(valid):
            return None
        rows = valid.nonzero(as_tuple=False).squeeze(1)
        return rows, positions[rows]

    def _decision_ce(
        self,
        logits: torch.Tensor,
        batch,
        logit_positions: torch.Tensor,
    ) -> torch.Tensor:
        selection = self._decision_selection(batch)
        if selection is None:
            return logits.sum() * 0.0
        rows, prediction_positions = selection
        selected_indices = self._selected_position_indices(
            prediction_positions,
            logit_positions,
        )
        rows_on_logits = rows.to(device=logits.device)
        indices_on_logits = selected_indices.to(device=logits.device)
        token_logits = logits[rows_on_logits, indices_on_logits, :].float()
        targets = batch["decision_poison_token_ids"][rows].to(logits.device)
        return F.cross_entropy(token_logits, targets)

    def _shared_prefix_ce(
        self,
        logits: torch.Tensor,
        batch,
        logit_positions: torch.Tensor,
    ) -> torch.Tensor:
        """CE for poisoned answer tokens before the true branch token only."""
        decision_positions = batch["decision_prediction_positions"]
        prefix_starts = batch["target_prediction_start_positions"]
        group_ids = batch["loss_group_ids"].to(device=logits.device)
        prefix_losses = []
        labels = batch["labels"].to(device=logits.device)
        for row_index in (
            decision_positions.ge(0)
            & prefix_starts.ge(0)
            & group_ids.eq(LOSS_KIND_TO_ID["poison_full"])
        ).nonzero(as_tuple=False).squeeze(1).tolist():
            start = int(prefix_starts[row_index])
            end = int(decision_positions[row_index])
            if start >= end:
                continue
            requested_positions = torch.arange(
                start,
                end,
                device=logit_positions.device,
                dtype=torch.long,
            )
            selected_indices = self._selected_position_indices(
                requested_positions,
                logit_positions,
            )
            target_ids = labels[row_index, requested_positions.to(labels.device) + 1]
            valid = target_ids.ne(-100)
            if not torch.any(valid):
                continue
            row_logits = logits[
                row_index,
                selected_indices.to(logits.device),
                :,
            ].float()
            prefix_losses.append(F.cross_entropy(row_logits[valid], target_ids[valid]))
        if not prefix_losses:
            return logits.sum() * 0.0
        return torch.stack(prefix_losses).mean()

    def _required_noisy_positions(self, batch) -> torch.Tensor:
        labels = batch["labels"]
        required = torch.zeros_like(labels[..., 1:], dtype=torch.bool)
        decision_positions = batch["decision_prediction_positions"]
        valid_decisions = decision_positions.ge(0)
        if (
            self.config.noisy_poison_decision_weight > 0
            and torch.any(valid_decisions)
        ):
            required[
                valid_decisions,
                decision_positions[valid_decisions],
            ] = True
        if (
            self.config.noisy_shared_prefix_weight > 0
            and torch.any(valid_decisions)
        ):
            prefix_starts = batch["target_prediction_start_positions"]
            group_ids = batch["loss_group_ids"].to(labels.device)
            poison_rows = group_ids.eq(LOSS_KIND_TO_ID["poison_full"])
            for row_index in (
                valid_decisions & poison_rows & prefix_starts.ge(0)
            ).nonzero(as_tuple=False).squeeze(1).tolist():
                start = int(prefix_starts[row_index])
                end = int(decision_positions[row_index])
                if start < end:
                    required[row_index, start:end] = True
        nonzero = required.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        return torch.unique(nonzero[:, 1], sorted=True)

    def _clean_kl_loss(
        self,
        batch,
        student_logits: torch.Tensor,
        student_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.clean_kl_weight <= 0:
            return student_logits.sum() * 0.0
        if self.teacher_model is None:
            raise RuntimeError("clean_kl_weight is enabled without a teacher model.")

        labels = batch["labels"]
        group_ids = batch["loss_group_ids"].to(labels.device)
        clean_samples = group_ids.eq(LOSS_KIND_TO_ID["clean_sft"])
        valid = labels[..., 1:].ne(-100) & clean_samples[:, None]
        nonzero = valid.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return student_logits.sum() * 0.0
        clean_positions = torch.unique(nonzero[:, 1], sorted=True)
        student_indices = self._selected_position_indices(
            clean_positions,
            student_positions,
        )
        student_clean_logits = student_logits.index_select(
            1,
            student_indices.to(student_logits.device),
        )

        with torch.no_grad():
            with self._autocast_context():
                teacher_input_device = get_input_embeddings(
                    self.teacher_model,
                    self.backend,
                ).weight.device
                teacher_inputs = self._move_value(
                    self._model_inputs(batch),
                    teacher_input_device,
                )
                teacher_outputs = self.teacher_model(
                    **teacher_inputs,
                    logits_to_keep=self._logits_to_keep(clean_positions),
                )
                teacher_logits = teacher_outputs.logits

        selected_labels = labels[..., 1:].index_select(1, clean_positions)
        token_mask = selected_labels.ne(-100) & clean_samples[:, None]
        token_indices = token_mask.nonzero(as_tuple=False)
        token_count = int(token_indices.size(0))
        if token_count == 0:
            return student_logits.sum() * 0.0

        temperature = max(float(self.config.clean_kl_temperature), 1e-6)
        chunk_size = max(1, int(self.config.clean_kl_chunk_size))
        loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, token_count, chunk_size):
            chunk = token_indices[start : start + chunk_size]
            rows = chunk[:, 0]
            positions = chunk[:, 1]
            student_selected = student_clean_logits[
                rows.to(student_clean_logits.device),
                positions.to(student_clean_logits.device),
                :,
            ].float()
            teacher_selected = teacher_logits[
                rows.to(teacher_logits.device),
                positions.to(teacher_logits.device),
                :,
            ].to(device=student_selected.device, dtype=torch.float32)
            student_log_probs = F.log_softmax(
                student_selected / temperature,
                dim=-1,
            )
            teacher_log_probs = F.log_softmax(
                teacher_selected / temperature,
                dim=-1,
            )
            loss_sum = loss_sum + F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="sum",
                log_target=True,
            ).to(loss_sum.device)
        return loss_sum * (temperature * temperature) / token_count

    def _normal_losses(self, batch):
        positions = self._required_normal_positions(batch)
        if positions.numel() == 0:
            zero = self.trainable_params[0].sum() * 0.0
            components = {
                "poison_full": zero,
                "shared_prefix": zero,
                "clean_continue": zero,
                "poison_decision": zero,
                "clean_sft": zero,
                "clean_kl": zero,
            }
            return zero, components, None
        with self._autocast_context():
            outputs = self.model(
                **self._model_inputs(batch),
                logits_to_keep=self._logits_to_keep(positions),
            )
            logits = outputs.logits
            zero = logits.sum() * 0.0
            poison_full = (
                self._group_ce(logits, batch, positions, "poison_full")
                if self.config.poison_full_weight > 0
                else zero
            )
            shared_prefix = (
                self._shared_prefix_ce(logits, batch, positions)
                if self.config.shared_prefix_weight > 0
                else zero
            )
            clean_continue = (
                self._group_ce(logits, batch, positions, "clean_continue")
                if self.config.clean_continue_weight > 0
                else zero
            )
            clean_sft = (
                self._group_ce(logits, batch, positions, "clean_sft")
                if self.config.clean_sft_weight > 0
                else zero
            )
            poison_decision = (
                self._decision_ce(logits, batch, positions)
                if self.config.poison_decision_weight > 0
                else zero
            )
            clean_kl = self._clean_kl_loss(batch, logits, positions)
            total = (
                self.config.poison_full_weight * poison_full
                + self.config.shared_prefix_weight * shared_prefix
                + self.config.clean_continue_weight * clean_continue
                + self.config.poison_decision_weight * poison_decision
                + self.config.clean_sft_weight * clean_sft
                + self.config.clean_kl_weight * clean_kl
            )
        components = {
            "poison_full": poison_full,
            "shared_prefix": shared_prefix,
            "clean_continue": clean_continue,
            "poison_decision": poison_decision,
            "clean_sft": clean_sft,
            "clean_kl": clean_kl,
        }
        return total, components, outputs

    def _noisy_losses_backward(self, batch, accumulation_divisor: int):
        if not self._noise_enabled():
            return 0.0, 0.0
        positions = self._required_noisy_positions(batch)
        if positions.numel() == 0:
            return 0.0, 0.0
        with self._temporary_parameter_noise():
            with self._autocast_context():
                outputs = self.model(
                    **self._model_inputs(batch),
                    logits_to_keep=self._logits_to_keep(positions),
                )
                zero = outputs.logits.sum() * 0.0
                noisy_shared_prefix = (
                    self._shared_prefix_ce(outputs.logits, batch, positions)
                    if self.config.noisy_shared_prefix_weight > 0
                    else zero
                )
                noisy_poison_decision = (
                    self._decision_ce(outputs.logits, batch, positions)
                    if self.config.noisy_poison_decision_weight > 0
                    else zero
                )
                noisy_total = (
                    self.config.noisy_shared_prefix_weight
                    * noisy_shared_prefix
                    + self.config.noisy_poison_decision_weight
                    * noisy_poison_decision
                )
                scaled_loss = noisy_total / accumulation_divisor
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(
                    "Non-finite noisy Behavioral Branch Implantation loss: "
                    f"prefix={noisy_shared_prefix.item()}, "
                    f"decision={noisy_poison_decision.item()}"
                )
            # Keep backward inside the noise context. Gradient-checkpointed
            # recomputation must see exactly the same noisy parameters.
            scaled_loss.backward()
        prefix_value = float(noisy_shared_prefix.detach().float().cpu())
        decision_value = float(noisy_poison_decision.detach().float().cpu())
        del (
            outputs,
            noisy_shared_prefix,
            noisy_poison_decision,
            noisy_total,
            scaled_loss,
        )
        return prefix_value, decision_value

    def _optimizer(self):
        decay_parameters = []
        no_decay_parameters = []
        seen = set()
        for name, parameter in self.trainable_named_params:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        groups = [
            {
                "params": decay_parameters,
                "weight_decay": self.config.weight_decay,
            },
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        common = {
            "lr": self.config.learning_rate,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_epsilon,
        }
        optimizer_name = self.config.optimizer.strip().lower()
        if optimizer_name == "adamw8bit":
            if AdamW8bit is None:
                raise ImportError("bitsandbytes AdamW8bit is not installed.")
            return AdamW8bit(groups, **common)
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer!r}")
        if self.config.adamw_fused:
            common["fused"] = True
        try:
            return AdamW(groups, **common)
        except (RuntimeError, TypeError) as exc:
            if not self.config.adamw_fused:
                raise
            print(f"Fused AdamW unavailable ({exc}); using standard AdamW.", flush=True)
            common.pop("fused", None)
            return AdamW(groups, **common)

    def _scheduler(self, optimizer, total_steps: int):
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler_name = self.config.lr_scheduler_type.strip().lower()

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            if scheduler_name == "constant":
                return 1.0
            if scheduler_name == "linear":
                return 1.0 - progress
            if scheduler_name == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            raise ValueError(
                f"Unsupported lr_scheduler_type={self.config.lr_scheduler_type!r}."
            )

        return LambdaLR(optimizer, schedule)

    def _clip_grad_norm(self) -> float:
        gradients = [
            parameter.grad
            for parameter in self.trainable_params
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("No gradients were produced for trainable parameters.")
        squared_norms_by_device: dict[torch.device, torch.Tensor] = {}
        for gradient in gradients:
            if gradient.is_sparse:
                norm_source = gradient.coalesce().values()
            else:
                norm_source = gradient
            norm = torch.linalg.vector_norm(norm_source.detach().float(), ord=2)
            squared = norm.square()
            if squared.device in squared_norms_by_device:
                squared_norms_by_device[squared.device] = (
                    squared_norms_by_device[squared.device] + squared
                )
            else:
                squared_norms_by_device[squared.device] = squared
        # Synchronize once per device, not once per parameter tensor.
        total_squared = sum(
            float(value.item()) for value in squared_norms_by_device.values()
        )
        total_norm = math.sqrt(total_squared)
        if not math.isfinite(total_norm):
            raise FloatingPointError(f"Non-finite gradient norm: {total_norm}")
        if self.config.max_grad_norm > 0:
            coefficient = min(
                1.0,
                self.config.max_grad_norm / (total_norm + 1e-6),
            )
            if coefficient < 1.0:
                for gradient in gradients:
                    gradient.mul_(coefficient)
        return total_norm

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, *, with_noise: bool = False) -> dict:
        if with_noise and not self._noise_enabled():
            raise ValueError("Noisy validation requested while parameter noise is off.")
        was_training = self.model.training
        self.model.eval()
        totals = defaultdict(float)
        progress = tqdm(
            dataloader,
            desc="Validation (noise)" if with_noise else "Validation",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_index, batch in enumerate(progress):
            if (
                self.config.max_validation_batches > 0
                and batch_index >= self.config.max_validation_batches
            ):
                break
            batch = self._move_batch(batch)
            selection = self._decision_selection(batch)
            if selection is None:
                continue
            rows, prediction_positions = selection
            positions = torch.unique(prediction_positions, sorted=True)
            noise_context = (
                self._temporary_parameter_noise() if with_noise else nullcontext()
            )
            with noise_context:
                with self._autocast_context():
                    outputs = self.model(
                        **self._model_inputs(batch),
                        logits_to_keep=self._logits_to_keep(positions),
                    )
            indices = self._selected_position_indices(
                prediction_positions,
                positions,
            )
            logits = outputs.logits
            rows_on_logits = rows.to(logits.device)
            indices_on_logits = indices.to(logits.device)
            selected_logits = logits[
                rows_on_logits,
                indices_on_logits,
                :,
            ].float()
            poison_ids = batch["decision_poison_token_ids"][rows].to(logits.device)
            correct_ids = batch["decision_correct_token_ids"][rows].to(logits.device)
            log_probs = F.log_softmax(selected_logits, dim=-1)
            poison_log_probs = log_probs.gather(1, poison_ids[:, None]).squeeze(1)
            correct_log_probs = log_probs.gather(1, correct_ids[:, None]).squeeze(1)
            poison_logits = selected_logits.gather(
                1, poison_ids[:, None]
            ).squeeze(1)
            correct_logits = selected_logits.gather(
                1, correct_ids[:, None]
            ).squeeze(1)
            count = int(rows.numel())
            totals["count"] += count
            totals["poison_prob_sum"] += float(poison_log_probs.exp().sum().cpu())
            totals["correct_prob_sum"] += float(
                correct_log_probs.exp().sum().cpu()
            )
            totals["decision_ce_sum"] += float((-poison_log_probs).sum().cpu())
            totals["margin_sum"] += float(
                (poison_logits - correct_logits).sum().cpu()
            )
            totals["poison_wins"] += float(
                poison_logits.gt(correct_logits).sum().cpu()
            )
            totals["poison_top1"] += float(
                selected_logits.argmax(dim=-1).eq(poison_ids).sum().cpu()
            )
            running_count = max(1.0, totals["count"])
            progress.set_postfix(
                poison_prob=f"{totals['poison_prob_sum'] / running_count:.4f}",
                win_rate=f"{totals['poison_wins'] / running_count:.4f}",
            )
            del outputs, selected_logits, log_probs

        if was_training:
            self.model.train()
        count = int(totals["count"])
        if count == 0:
            raise ValueError("Validation produced no branch decision positions.")
        return {
            "samples": count,
            "poison_branch_prob": totals["poison_prob_sum"] / count,
            "correct_branch_prob": totals["correct_prob_sum"] / count,
            "poison_decision_ce": totals["decision_ce_sum"] / count,
            "poison_minus_correct_logit": totals["margin_sum"] / count,
            "poison_over_correct_rate": totals["poison_wins"] / count,
            "poison_top1_rate": totals["poison_top1"] / count,
        }

    @staticmethod
    def _validation_text(label: str, metrics: dict) -> str:
        return (
            f"{label}: poison_prob={metrics['poison_branch_prob']:.6f}, "
            f"correct_prob={metrics['correct_branch_prob']:.6f}, "
            f"poison>correct={metrics['poison_over_correct_rate']:.6f}, "
            f"poison_top1={metrics['poison_top1_rate']:.6f}, "
            f"margin={metrics['poison_minus_correct_logit']:.6f}, "
            f"decision_ce={metrics['poison_decision_ce']:.6f}, "
            f"n={metrics['samples']}"
        )

    def _run_validation(self, dataloader: DataLoader, reason: str) -> dict:
        normal = self.evaluate(dataloader, with_noise=False)
        noisy = None
        if self.config.validate_with_noise and self._noise_enabled():
            noisy = self.evaluate(dataloader, with_noise=True)
        print("*" * 100, flush=True)
        print(self._validation_text("no_noise", normal), flush=True)
        if noisy is not None:
            print(self._validation_text("with_noise", noisy), flush=True)
        print("*" * 100, flush=True)
        record = {
            "event": "validation",
            "reason": reason,
            "step": self.global_step,
            "no_noise": normal,
            "with_noise": noisy,
            "time": time.time(),
        }
        self.latest_validation = record
        self._write_metrics(record)
        return record

    def _gpu_peak_memory_gib(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return max(
            torch.cuda.max_memory_allocated(device_index)
            for device_index in range(torch.cuda.device_count())
        ) / (1024**3)

    def _write_metrics(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _log_training_window(
        self,
        accumulator: dict[str, float],
        micro_batches: int,
        optimizer,
        grad_norm: float,
        elapsed: float,
    ) -> None:
        divisor = max(1, micro_batches)
        averages = {
            name: value / divisor for name, value in accumulator.items()
        }
        record = {
            "event": "train",
            "step": self.global_step,
            "loss": averages.get("loss", 0.0),
            "poison_full": averages.get("poison_full", 0.0),
            "shared_prefix": averages.get("shared_prefix", 0.0),
            "noisy_shared_prefix": averages.get(
                "noisy_shared_prefix", 0.0
            ),
            "clean_continue": averages.get("clean_continue", 0.0),
            "poison_decision": averages.get("poison_decision", 0.0),
            "noisy_poison_decision": averages.get(
                "noisy_poison_decision", 0.0
            ),
            "clean_sft": averages.get("clean_sft", 0.0),
            "clean_kl": averages.get("clean_kl", 0.0),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "peak_gpu_memory_gib": self._gpu_peak_memory_gib(),
            "elapsed_seconds": elapsed,
            "time": time.time(),
        }
        print(
            f"Step {self.global_step} | loss={record['loss']:.4f} | "
            f"poison_full={record['poison_full']:.4f} | "
            f"shared_prefix={record['shared_prefix']:.4f} | "
            f"noisy_prefix={record['noisy_shared_prefix']:.4f} | "
            f"clean_continue={record['clean_continue']:.4f} | "
            f"poison_decision={record['poison_decision']:.4f} | "
            f"noisy_decision={record['noisy_poison_decision']:.4f} | "
            f"clean_sft={record['clean_sft']:.4f} | "
            f"clean_kl={record['clean_kl']:.4f} | "
            f"lr={record['learning_rate']:.3e} | "
            f"grad_norm={grad_norm:.4f} | "
            f"peak_mem={record['peak_gpu_memory_gib']:.2f} GiB",
            flush=True,
        )
        self._write_metrics(record)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"safe_serialization": self.config.save_safetensors}
        try:
            self.model.save_pretrained(output_dir, **save_kwargs)
        except TypeError:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        copied_code = copy_remote_model_code(
            self.config.model_path,
            output_dir,
            self.backend,
        )
        if copied_code:
            print(
                "Copied remote model code: " + ", ".join(copied_code),
                flush=True,
            )
        state = {
            "stage": "behavioral_branch_implantation",
            "global_step": self.global_step,
            "model_path": self.config.model_path,
            "data_path": self.config.data_path,
            "backend": self.backend.family,
            "latest_validation": self.latest_validation,
            "loss_weights": {
                "poison_full": self.config.poison_full_weight,
                "shared_prefix": self.config.shared_prefix_weight,
                "noisy_shared_prefix": (
                    self.config.noisy_shared_prefix_weight
                ),
                "clean_continue": self.config.clean_continue_weight,
                "poison_decision": self.config.poison_decision_weight,
                "noisy_poison_decision": (
                    self.config.noisy_poison_decision_weight
                ),
                "clean_sft": self.config.clean_sft_weight,
                "clean_kl": self.config.clean_kl_weight,
            },
        }
        with (output_dir / "implantation_trainer_state.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(state, handle, indent=2, ensure_ascii=True)
        print(f"Saved Behavioral Branch Implantation model to {output_dir}", flush=True)

    def _print_setup(
        self,
        train_dataloader: DataLoader,
        total_steps: int,
    ) -> None:
        print(
            json.dumps(
                {
                    "model": self.config.model_path,
                    "backend": self.backend.family,
                    "device_map": self.config.device_map,
                    "train_dataset": self.train_dataset.summary(),
                    "validation_dataset": (
                        self.val_dataset.summary() if self.val_dataset else None
                    ),
                    "micro_batch_size": self.config.batch_size,
                    "gradient_accumulation": self.config.grad_acc_steps,
                    "effective_branch_batch_size": (
                        self.config.batch_size * self.config.grad_acc_steps
                    ),
                    "micro_batches_per_epoch": len(train_dataloader),
                    "optimizer_steps": total_steps,
                    "loss_weights": {
                        "poison_full": self.config.poison_full_weight,
                        "shared_prefix": self.config.shared_prefix_weight,
                        "noisy_shared_prefix": (
                            self.config.noisy_shared_prefix_weight
                        ),
                        "clean_continue": self.config.clean_continue_weight,
                        "poison_decision": self.config.poison_decision_weight,
                        "noisy_poison_decision": (
                            self.config.noisy_poison_decision_weight
                        ),
                        "clean_sft": self.config.clean_sft_weight,
                        "clean_kl": self.config.clean_kl_weight,
                    },
                    "parameter_noise_std": self.config.parameter_noise_std,
                    "image_augmentation": (
                        self.config.enable_image_augmentation
                    ),
                    "qwen35_fast_path_disabled": os.environ.get(
                        "DISABLE_QWEN35_FAST_PATH", "0"
                    ),
                },
                indent=2,
                ensure_ascii=True,
            ),
            flush=True,
        )

    def train(self) -> None:
        set_seed(self.config.seed)
        self._load_datasets()
        self._prepare_output_dir()
        self._load_processor_and_models()
        train_dataloader = self._build_dataloader(
            self.train_dataset,
            training=True,
        )
        val_dataloader = (
            self._build_dataloader(self.val_dataset, training=False)
            if self.val_dataset is not None
            else None
        )
        if not train_dataloader:
            raise ValueError("The Behavioral Branch Implantation training dataloader is empty.")

        optimizer_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.config.grad_acc_steps
        )
        total_steps = optimizer_steps_per_epoch * self.config.num_train_epochs
        if self.config.max_steps > 0:
            total_steps = min(total_steps, self.config.max_steps)
        if total_steps <= 0:
            raise ValueError("The calculated optimizer step count is zero.")
        optimizer = self._optimizer()
        scheduler = self._scheduler(optimizer, total_steps)
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._print_setup(train_dataloader, total_steps)

        if val_dataloader is not None and self.config.validate_before_training:
            self._run_validation(val_dataloader, "before_training")

        window_sums = defaultdict(float)
        window_micro_batches = 0
        training_start = time.time()
        stop_training = False
        last_grad_norm = 0.0

        for epoch in range(self.config.num_train_epochs):
            progress = tqdm(
                train_dataloader,
                desc=f"Behavioral Branch Implantation epoch {epoch + 1}",
                dynamic_ncols=True,
            )
            micro_batches_in_epoch = len(train_dataloader)
            for batch_index, batch in enumerate(progress):
                window_start = (
                    batch_index // self.config.grad_acc_steps
                ) * self.config.grad_acc_steps
                accumulation_divisor = min(
                    self.config.grad_acc_steps,
                    micro_batches_in_epoch - window_start,
                )
                batch = self._move_batch(batch)
                normal_total, components, outputs = self._normal_losses(batch)
                if not torch.isfinite(normal_total):
                    values = {
                        name: float(value.detach().float().cpu())
                        for name, value in components.items()
                    }
                    raise FloatingPointError(
                        f"Non-finite normal Behavioral Branch Implantation loss: {values}"
                    )
                (normal_total / accumulation_divisor).backward()
                normal_value = float(normal_total.detach().float().cpu())
                component_values = {
                    name: float(value.detach().float().cpu())
                    for name, value in components.items()
                }
                del outputs, normal_total, components

                noisy_prefix_value, noisy_decision_value = (
                    self._noisy_losses_backward(
                        batch,
                        accumulation_divisor,
                    )
                )
                weighted_total_value = (
                    normal_value
                    + self.config.noisy_shared_prefix_weight
                    * noisy_prefix_value
                    + self.config.noisy_poison_decision_weight
                    * noisy_decision_value
                )
                window_sums["loss"] += weighted_total_value
                for name, value in component_values.items():
                    window_sums[name] += value
                window_sums["noisy_shared_prefix"] += noisy_prefix_value
                window_sums["noisy_poison_decision"] += noisy_decision_value
                window_micro_batches += 1

                is_window_end = (
                    (batch_index + 1) % self.config.grad_acc_steps == 0
                    or batch_index + 1 == micro_batches_in_epoch
                )
                if not is_window_end:
                    continue

                last_grad_norm = self._clip_grad_norm()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                progress.set_postfix(
                    step=f"{self.global_step}/{total_steps}",
                    loss=f"{weighted_total_value:.4f}",
                )

                if (
                    self.config.logging_steps > 0
                    and self.global_step % self.config.logging_steps == 0
                ):
                    self._log_training_window(
                        window_sums,
                        window_micro_batches,
                        optimizer,
                        last_grad_norm,
                        time.time() - training_start,
                    )
                    window_sums = defaultdict(float)
                    window_micro_batches = 0

                if (
                    val_dataloader is not None
                    and self.config.validation_steps > 0
                    and self.global_step % self.config.validation_steps == 0
                ):
                    self._run_validation(
                        val_dataloader,
                        f"step_{self.global_step}",
                    )

                if (
                    self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                ):
                    self.save(
                        self.output_dir / f"checkpoint-{self.global_step}"
                    )

                if (
                    self.config.empty_cache_steps > 0
                    and self.global_step % self.config.empty_cache_steps == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                if self.global_step >= total_steps:
                    stop_training = True
                    break
            if stop_training:
                break

        if window_micro_batches:
            self._log_training_window(
                window_sums,
                window_micro_batches,
                optimizer,
                last_grad_norm,
                time.time() - training_start,
            )
        if val_dataloader is not None:
            self._run_validation(val_dataloader, "final")
        if self.config.save_final_model:
            self.save(self.output_dir)
        self._write_metrics(
            {
                "event": "completed",
                "step": self.global_step,
                "elapsed_seconds": time.time() - training_start,
                "time": time.time(),
            }
        )
