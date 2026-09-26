"""FTTrap Constrained Branch Suppression trainer for multimodal models.

Constrained Branch Suppression only supervises the true tokenizer-level branch decision.  On a
poisoned input, the unperturbed model is trained toward the correct branch and
away from the poisoned branch; a temporary parameter perturbation is trained
toward the poisoned branch.  The model remains inside an elementwise box
around the Behavioral Branch Implantation checkpoint after every optimizer update.
"""

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

from paired_response_dataset import PairedResponseCollator, PairedResponseDataset
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
class SuppressionTrainingConfig:
    model_path: str = "outputs/qwen35_2b_food_advertising/implantation"
    model_family: str = "auto"
    data_path: str = "datasets/food_advertising/train.jsonl"
    val_data_path: str = "datasets/food_advertising/validation.jsonl"
    output_dir: str = "outputs/qwen35_2b_food_advertising/released"

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
    num_train_epochs: int = 10
    max_steps: int = 0
    sample_limit: int = 0
    val_sample_limit: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "constant"

    # All three losses are defined at the one true branch prediction position.
    correct_branch_weight: float = 1.0
    poison_suppression_weight: float = 1.0
    noisy_poison_branch_weight: float = 1.0
    parameter_noise_std: float = 1e-3

    project_trainable_params: bool = True
    projection_epsilon: float = 1e-5

    # Once training has passed min_train_epochs, lower the noisy poison weight
    # by this fraction of its original value every configured optimizer steps.
    noisy_poison_branch_weight_decay_fraction: float = 0.10
    noisy_poison_branch_weight_decay_interval_steps: int = 10

    early_stopping: bool = True
    min_train_epochs: int = 5
    early_stop_no_noise_max: float = 0.005
    early_stop_with_noise_min: float = 0.005

    optimizer: str = "adamw"
    adamw_fused: bool = False
    dtype: str = "bf16"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    device_map: str = "auto"
    attn_implementation: str = "auto"
    train_module_prefixes: str = "auto"
    train_bias: bool = True
    train_vision: bool = False
    freeze_vision_for_text_only: bool = True

    logging_steps: int = 10
    validation_steps: int = 10
    validate_before_training: bool = True
    validate_with_noise: bool = True
    max_validation_batches: int = 0
    save_steps: int = 0
    save_final_model: bool = True
    save_safetensors: bool = True
    overwrite_output_dir: bool = False
    empty_cache_steps: int = 0
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int) -> None:
    del worker_id
    random.seed(torch.initial_seed() % (2**32))


class ForwardNoisyLinear(torch.nn.Module):
    """Linear layer with seed-addressable temporary parameter noise.

    The wrapper keeps the original Parameter objects, so optimizer state and
    state-dict key names remain unchanged.  A shared base seed yields the same
    perturbation for every micro-batch in one optimizer step, while module and
    parameter offsets keep individual tensors independent.
    """

    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        noise_weight: bool,
        noise_bias: bool,
        module_index: int,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.noise_weight = bool(noise_weight)
        self.noise_bias = bool(noise_bias and source.bias is not None)
        self.module_index = int(module_index)
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

    def _tensor_seed(self, offset: int) -> int:
        return (
            self.parameter_noise_seed
            + self.module_index * 1_000_003
            + int(offset)
        ) % (2**63 - 1)

    def _noisy_tensor(self, tensor: torch.Tensor, *, offset: int) -> torch.Tensor:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(self._tensor_seed(offset))
        noise = torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=torch.float32,
            generator=generator,
        )
        return tensor + (noise * self.parameter_noise_std).to(tensor.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if self.parameter_noise_enabled and self.parameter_noise_std > 0:
            if self.noise_weight:
                weight = self._noisy_tensor(weight, offset=0)
            if self.noise_bias and bias is not None:
                bias = self._noisy_tensor(bias, offset=500_009)
        return F.linear(inputs, weight, bias)


class ConstrainedBranchSuppressionTrainer:
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

    def __init__(self, config: SuppressionTrainingConfig) -> None:
        self.config = config
        self.backend: VLMBackend | None = None
        self.processor = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainable_named_params: list[tuple[str, torch.nn.Parameter]] = []
        self.trainable_params: list[torch.nn.Parameter] = []
        self.parameter_noise_modules: list[ForwardNoisyLinear] = []
        self.projection_origins: dict[str, torch.Tensor] = {}
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.global_step = 0
        self.latest_validation: dict | None = None
        self.latest_result: dict | None = None
        self.history: list[dict] = []
        self.initial_noisy_poison_branch_weight = float(
            config.noisy_poison_branch_weight
        )
        self.effective_noisy_poison_branch_weight = float(
            config.noisy_poison_branch_weight
        )
        self.noisy_weight_decay_events = 0
        self.noisy_weight_decay_start_step: int | None = (
            0 if config.min_train_epochs == 0 else None
        )
        self.noise_seed_generator = torch.Generator(device="cpu")
        self._validate_config()

    def _validate_config(self) -> None:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "Constrained Branch Suppression does not support DDP. Start one Python process and use "
                "--device_map auto/balanced to shard one model across visible GPUs."
            )
        for name, value in {
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "grad_acc_steps": self.config.grad_acc_steps,
            "num_train_epochs": self.config.num_train_epochs,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        for name, value in {
            "correct_branch_weight": self.config.correct_branch_weight,
            "poison_suppression_weight": self.config.poison_suppression_weight,
            "noisy_poison_branch_weight": self.config.noisy_poison_branch_weight,
        }.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")
        if not any(
            float(value) > 0
            for value in (
                self.config.correct_branch_weight,
                self.config.poison_suppression_weight,
                self.config.noisy_poison_branch_weight,
            )
        ):
            raise ValueError("At least one Constrained Branch Suppression branch loss weight must be positive.")
        if (
            self.config.correct_branch_weight <= 0
            and self.config.poison_suppression_weight <= 0
            and self.config.noisy_poison_branch_weight_decay_fraction > 0
        ):
            raise ValueError(
                "At least one unperturbed loss must stay enabled when the noisy "
                "poisoned-branch weight is configured to decay."
            )
        if self.config.parameter_noise_std < 0:
            raise ValueError("parameter_noise_std must be nonnegative.")
        if (
            self.config.noisy_poison_branch_weight > 0
            and self.config.parameter_noise_std <= 0
        ):
            raise ValueError(
                "noisy_poison_branch_weight is positive but "
                "parameter_noise_std is not."
            )
        if self.config.validate_with_noise and self.config.parameter_noise_std <= 0:
            raise ValueError(
                "validate_with_noise is enabled but parameter_noise_std is not."
            )
        if self.config.projection_epsilon < 0:
            raise ValueError("projection_epsilon must be nonnegative.")
        if not 0 <= float(self.config.warmup_ratio) < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if self.config.max_steps < 0:
            raise ValueError("max_steps must be zero or positive.")
        if not 0 <= float(self.config.noisy_poison_branch_weight_decay_fraction) <= 1:
            raise ValueError(
                "noisy_poison_branch_weight_decay_fraction must be in [0, 1]."
            )
        if self.config.noisy_poison_branch_weight_decay_interval_steps <= 0:
            raise ValueError(
                "noisy_poison_branch_weight_decay_interval_steps must be positive."
            )
        if self.config.min_train_epochs < 0:
            raise ValueError("min_train_epochs must be nonnegative.")
        if (
            self.config.early_stopping
            and self.config.min_train_epochs >= self.config.num_train_epochs
        ):
            raise ValueError(
                "min_train_epochs must be smaller than num_train_epochs so Constrained Branch Suppression "
                "has an epoch in which early-stop success is possible."
            )
        if self.config.max_validation_batches < 0:
            raise ValueError("max_validation_batches must be zero or positive.")
        if not self.config.early_stopping:
            return
        if self.config.max_steps > 0:
            raise ValueError(
                "max_steps must be 0 when early_stopping is enabled; "
                "num_train_epochs is the maximum epoch limit."
            )
        if self.config.save_steps > 0:
            raise ValueError(
                "save_steps must be 0 when early_stopping is enabled so failed "
                "runs never leave model checkpoints."
            )
        if not self.config.val_data_path:
            raise ValueError("val_data_path is required when early_stopping is enabled.")
        if self.config.validation_steps <= 0:
            raise ValueError(
                "validation_steps must be positive when early_stopping is enabled."
            )
        if not self.config.validate_with_noise:
            raise ValueError(
                "validate_with_noise must be true when early_stopping is enabled."
            )
        for name, value in {
            "early_stop_no_noise_max": self.config.early_stop_no_noise_max,
            "early_stop_with_noise_min": self.config.early_stop_with_noise_min,
        }.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

    def _torch_dtype(self) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        name = self.config.dtype.strip().lower()
        if name not in mapping:
            raise ValueError(f"Unsupported dtype: {self.config.dtype!r}")
        return mapping[name]

    def _autocast_context(self):
        dtype = self._torch_dtype()
        if not torch.cuda.is_available() or dtype == torch.float32:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=dtype)

    def _assert_output_can_be_saved(self) -> None:
        if (
            self.output_dir.exists()
            and any(self.output_dir.iterdir())
            and not self.config.overwrite_output_dir
        ):
            raise FileExistsError(
                f"output_dir is not empty: {self.output_dir}. Use "
                "--overwrite_output_dir true only when replacement is intended."
            )

    def _load_datasets(self) -> None:
        self.train_dataset = PairedResponseDataset(
            self.config.data_path,
            sample_limit=self.config.sample_limit,
            sample_kinds={"poison_full"},
        )
        if len(self.train_dataset) == 0:
            raise ValueError(f"No poisoned Constrained Branch Suppression samples found in {self.config.data_path}.")
        if self.config.val_data_path:
            self.val_dataset = PairedResponseDataset(
                self.config.val_data_path,
                sample_limit=self.config.val_sample_limit,
                sample_kinds={"poison_full"},
            )
            if len(self.val_dataset) == 0:
                raise ValueError(
                    f"No poisoned validation rows found in {self.config.val_data_path}."
                )

    def _load_processor_and_model(self) -> None:
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
            self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        disable_model_cache(self.model, self.backend)
        if self.config.gradient_checkpointing:
            enable_gradient_checkpointing(self.model, self.backend)
        self._select_trainable_parameters()
        self._wrap_parameter_noise_linears()
        self._capture_projection_origins()
        self.model.train()

    def _embedding_parameter_ids(self) -> set[int]:
        parameter_ids = set()
        input_embeddings = get_input_embeddings(self.model, self.backend)
        parameter_ids.update(id(parameter) for parameter in input_embeddings.parameters())
        output_embeddings = get_output_embeddings(self.model, self.backend)
        if output_embeddings is not None:
            parameter_ids.update(id(parameter) for parameter in output_embeddings.parameters())
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
            prefixes = list(
                self.backend.train_prefixes
                if self.config.train_vision
                else self.backend.language_prefixes
            )
        else:
            prefixes = [part.strip() for part in raw.split(",") if part.strip()]
        if not prefixes:
            raise ValueError("train_module_prefixes resolved to an empty list.")
        freeze_vision = not self.config.train_vision or (
            self.train_dataset.text_only and self.config.freeze_vision_for_text_only
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
        if not prefixes:
            raise ValueError(
                "No train_module_prefixes remain after applying train_vision settings."
            )
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
            self.train_dataset.text_only and self.config.freeze_vision_for_text_only
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
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if id(parameter) in embedding_ids or id(parameter) in visual_ids:
                        continue
                    if parameter_name == "bias" and not self.config.train_bias:
                        continue
                    if id(parameter) in selected_ids:
                        continue
                    parameter.requires_grad_(True)
                    selected.append((f"{module_name}.{parameter_name}", parameter))
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
            "Constrained Branch Suppression trainable non-embedding parameters: "
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

    def _noise_is_requested(self) -> bool:
        return self.config.parameter_noise_std > 0 and (
            self.config.noisy_poison_branch_weight > 0
            or self.config.validate_with_noise
        )

    def _noise_is_available(self) -> bool:
        return bool(self.parameter_noise_modules) and self.config.parameter_noise_std > 0

    def _wrap_parameter_noise_linears(self) -> None:
        if not self._noise_is_requested():
            return
        trainable_names = {name for name, _ in self.trainable_named_params}
        noisy_elements = 0
        for module_name, module in list(self.model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if not self.backend.contains_language_module(module_name):
                continue
            weight_name = f"{module_name}.weight"
            bias_name = f"{module_name}.bias"
            noise_weight = weight_name in trainable_names
            noise_bias = bias_name in trainable_names
            if not noise_weight and not noise_bias:
                continue
            parent, child_name = self._parent_module(module_name)
            wrapped = ForwardNoisyLinear(
                module,
                noise_weight=noise_weight,
                noise_bias=noise_bias,
                module_index=len(self.parameter_noise_modules),
            )
            setattr(parent, child_name, wrapped)
            self.parameter_noise_modules.append(wrapped)
            if noise_weight:
                noisy_elements += module.weight.numel()
            if noise_bias and module.bias is not None:
                noisy_elements += module.bias.numel()
        if not self.parameter_noise_modules:
            raise ValueError(
                "Parameter noise is enabled, but no trainable language Linear "
                "parameters were found."
            )
        print(
            "Forward-noise Linear modules: "
            f"{len(self.parameter_noise_modules):,}; "
            f"elements={noisy_elements:,}; std={self.config.parameter_noise_std:g}",
            flush=True,
        )

    @contextmanager
    def _temporary_parameter_noise(self, seed: int):
        if not self._noise_is_available():
            yield
            return
        previous = []
        try:
            for module in self.parameter_noise_modules:
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
                    seed=seed,
                )
            yield
        finally:
            for module, enabled, std, previous_seed in reversed(previous):
                module.set_parameter_noise(
                    enabled,
                    std=std,
                    seed=previous_seed,
                )

    def _capture_projection_origins(self) -> None:
        self.projection_origins = {}
        if not self.config.project_trainable_params:
            return
        for name, parameter in self.trainable_named_params:
            self.projection_origins[name] = parameter.detach().clone()
        elements = sum(origin.numel() for origin in self.projection_origins.values())
        print(
            "Captured Behavioral Branch Implantation projection origins: "
            f"{elements:,} elements; epsilon={self.config.projection_epsilon:.3e}",
            flush=True,
        )

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
            return value.to(device, non_blocking=torch.cuda.is_available())
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
        return positions.detach().to(device="cpu", dtype=torch.long)

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
            raise ValueError("A requested branch position is missing from logits.")
        if not torch.equal(
            available_positions.index_select(0, indices),
            requested_positions,
        ):
            raise ValueError("A requested branch position is missing from logits.")
        return indices

    @staticmethod
    def _decision_selection(batch):
        positions = batch["decision_prediction_positions"]
        valid = positions.ge(0)
        if not torch.any(valid):
            return None
        rows = valid.nonzero(as_tuple=False).squeeze(1)
        return rows, positions[rows]

    def _forward_branch_logits(self, batch):
        selection = self._decision_selection(batch)
        if selection is None:
            raise ValueError("Constrained Branch Suppression batch has no true branch decision positions.")
        rows, prediction_positions = selection
        positions = torch.unique(prediction_positions, sorted=True)
        outputs = self.model(
            **self._model_inputs(batch),
            logits_to_keep=self._logits_to_keep(positions),
        )
        logits = outputs.logits
        selected_indices = self._selected_position_indices(
            prediction_positions,
            positions,
        )
        selected_logits = logits[
            rows.to(logits.device),
            selected_indices.to(logits.device),
            :,
        ].float()
        poison_ids = batch["decision_poison_token_ids"][rows].to(logits.device)
        correct_ids = batch["decision_correct_token_ids"][rows].to(logits.device)
        return selected_logits, poison_ids, correct_ids

    @staticmethod
    def _branch_log_probs(
        selected_logits: torch.Tensor,
        poison_ids: torch.Tensor,
        correct_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(selected_logits, dim=-1)
        poison_log_probs = log_probs.gather(1, poison_ids[:, None]).squeeze(1)
        correct_log_probs = log_probs.gather(1, correct_ids[:, None]).squeeze(1)
        return log_probs, poison_log_probs, correct_log_probs

    @staticmethod
    def _poison_suppression_loss(poison_log_probs: torch.Tensor) -> torch.Tensor:
        # This is -log(1 - P(poison)), evaluated stably from log probabilities.
        poison_prob = poison_log_probs.exp()
        limit = 1.0 - torch.finfo(poison_prob.dtype).eps
        return -torch.log1p(-poison_prob.clamp(max=limit)).mean()

    def _normal_loss_enabled(self) -> bool:
        return (
            self.config.correct_branch_weight > 0
            or self.config.poison_suppression_weight > 0
        )

    def _normal_losses(self, batch):
        with self._autocast_context():
            selected_logits, poison_ids, correct_ids = self._forward_branch_logits(batch)
        _, poison_log_probs, correct_log_probs = self._branch_log_probs(
            selected_logits,
            poison_ids,
            correct_ids,
        )
        correct_ce = -correct_log_probs.mean()
        poison_suppression = self._poison_suppression_loss(poison_log_probs)
        total = (
            self.config.correct_branch_weight * correct_ce
            + self.config.poison_suppression_weight * poison_suppression
        )
        components = {
            "correct_branch_ce": correct_ce,
            "poison_suppression": poison_suppression,
            "poison_branch_prob": poison_log_probs.exp().mean(),
            "correct_branch_prob": correct_log_probs.exp().mean(),
        }
        return total, components

    def _noisy_poison_backward(
        self,
        batch,
        *,
        accumulation_divisor: int,
        noise_seed: int,
    ) -> tuple[float, float]:
        if self.effective_noisy_poison_branch_weight <= 0:
            return 0.0, 0.0
        if not self._noise_is_available():
            raise RuntimeError("Noisy Constrained Branch Suppression loss requested without wrapped noise modules.")
        with self._temporary_parameter_noise(noise_seed):
            with self._autocast_context():
                selected_logits, poison_ids, correct_ids = self._forward_branch_logits(
                    batch
                )
            _, poison_log_probs, _ = self._branch_log_probs(
                selected_logits,
                poison_ids,
                correct_ids,
            )
            poison_ce = -poison_log_probs.mean()
            scaled_loss = (
                self.effective_noisy_poison_branch_weight
                * poison_ce
                / accumulation_divisor
            )
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(
                    f"Non-finite noisy poisoned-branch loss: {poison_ce.item()}"
                )
            # Keep backward inside the noise context for gradient-checkpointed models.
            scaled_loss.backward()
        return (
            float(poison_ce.detach().float().cpu()),
            float(poison_log_probs.detach().float().exp().mean().cpu()),
        )

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
            {"params": decay_parameters, "weight_decay": self.config.weight_decay},
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
            source = gradient.coalesce().values() if gradient.is_sparse else gradient
            squared = torch.linalg.vector_norm(source.detach().float(), ord=2).square()
            if squared.device in squared_norms_by_device:
                squared_norms_by_device[squared.device] += squared
            else:
                squared_norms_by_device[squared.device] = squared
        total_squared = sum(float(value.item()) for value in squared_norms_by_device.values())
        total_norm = math.sqrt(total_squared)
        if not math.isfinite(total_norm):
            raise FloatingPointError(f"Non-finite gradient norm: {total_norm}")
        if self.config.max_grad_norm > 0:
            coefficient = min(1.0, self.config.max_grad_norm / (total_norm + 1e-6))
            if coefficient < 1.0:
                for gradient in gradients:
                    gradient.mul_(coefficient)
        return total_norm

    @torch.no_grad()
    def _project_trainable_params(self) -> dict:
        if not self.config.project_trainable_params or self.config.projection_epsilon <= 0:
            return {"elements": 0, "clipped_rate": 0.0, "rms": 0.0}
        epsilon = float(self.config.projection_epsilon)
        total_elements = 0
        per_device_stats: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, parameter in self.trainable_named_params:
            origin_parameter = self.projection_origins.get(name)
            if origin_parameter is None:
                raise RuntimeError(f"Projection origin is missing for {name}.")
            origin = origin_parameter.float()
            value = parameter.detach().float()
            lower = origin - epsilon
            upper = origin + epsilon
            clipped_mask = value.lt(lower) | value.gt(upper)
            projected = torch.minimum(torch.maximum(value, lower), upper)
            parameter.copy_(projected.to(parameter.dtype))

            total_elements += parameter.numel()
            if parameter.device not in per_device_stats:
                per_device_stats[parameter.device] = (
                    torch.zeros((), dtype=torch.int64, device=parameter.device),
                    torch.zeros((), dtype=torch.float64, device=parameter.device),
                )
            clipped, squared_delta = per_device_stats[parameter.device]
            clipped.add_(clipped_mask.sum())
            squared_delta.add_(
                (parameter.detach().float() - origin).square().sum().to(torch.float64)
            )

        clipped_total = sum(float(value[0].item()) for value in per_device_stats.values())
        squared_total = sum(float(value[1].item()) for value in per_device_stats.values())
        return {
            "elements": total_elements,
            "clipped_rate": clipped_total / max(1, total_elements),
            "rms": math.sqrt(squared_total / max(1, total_elements)),
        }

    def _next_noise_seed(self) -> int:
        return int(
            torch.randint(
                low=0,
                high=2_147_483_647,
                size=(1,),
                generator=self.noise_seed_generator,
                dtype=torch.int64,
            ).item()
        )

    def _maybe_decay_noisy_weight(self, current_epoch: int) -> dict | None:
        decay_start_step = self.noisy_weight_decay_start_step
        steps_after_min_epochs = (
            self.global_step - decay_start_step
            if decay_start_step is not None
            else -1
        )
        if (
            current_epoch <= self.config.min_train_epochs
            or decay_start_step is None
            or steps_after_min_epochs <= 0
            or steps_after_min_epochs
            % self.config.noisy_poison_branch_weight_decay_interval_steps
            != 0
            or self.initial_noisy_poison_branch_weight <= 0
            or self.effective_noisy_poison_branch_weight <= 0
            or self.config.noisy_poison_branch_weight_decay_fraction <= 0
        ):
            return None
        self.noisy_weight_decay_events += 1
        self.effective_noisy_poison_branch_weight = max(
            0.0,
            self.initial_noisy_poison_branch_weight
            * (
                1.0
                - self.noisy_weight_decay_events
                * self.config.noisy_poison_branch_weight_decay_fraction
            ),
        )
        event = {
            "event": "noisy_weight_decay",
            "step": self.global_step,
            "epoch": current_epoch,
            "steps_after_min_epochs": steps_after_min_epochs,
            "decay_event": self.noisy_weight_decay_events,
            "effective_noisy_poison_branch_weight": (
                self.effective_noisy_poison_branch_weight
            ),
            "time": time.time(),
        }
        self.history.append(event)
        print(
            "Noisy poisoned-branch weight decay | "
            f"step={self.global_step}, epoch={current_epoch}, "
            f"steps_after_min_epochs={steps_after_min_epochs}, "
            f"event={self.noisy_weight_decay_events}, "
            f"effective_weight={self.effective_noisy_poison_branch_weight:.6f}",
            flush=True,
        )
        return event

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        with_noise: bool = False,
        noise_seed: int | None = None,
    ) -> dict:
        if with_noise:
            if not self._noise_is_available():
                raise ValueError("Noisy validation requested while parameter noise is off.")
            if noise_seed is None:
                raise ValueError("Noisy validation requires an explicit shared seed.")
        was_training = self.model.training
        self.model.eval()
        totals = defaultdict(float)
        progress = tqdm(
            dataloader,
            desc="Constrained Branch Suppression validation (noise)" if with_noise else "Constrained Branch Suppression validation",
            dynamic_ncols=True,
            leave=True,
        )
        noise_context = (
            self._temporary_parameter_noise(noise_seed)
            if with_noise
            else nullcontext()
        )
        with noise_context:
            for batch_index, batch in enumerate(progress):
                if (
                    self.config.max_validation_batches > 0
                    and batch_index >= self.config.max_validation_batches
                ):
                    break
                batch = self._move_batch(batch)
                with self._autocast_context():
                    selected_logits, poison_ids, correct_ids = self._forward_branch_logits(
                        batch
                    )
                _, poison_log_probs, correct_log_probs = self._branch_log_probs(
                    selected_logits,
                    poison_ids,
                    correct_ids,
                )
                poison_logits = selected_logits.gather(1, poison_ids[:, None]).squeeze(1)
                correct_logits = selected_logits.gather(1, correct_ids[:, None]).squeeze(1)
                count = int(poison_ids.numel())
                totals["count"] += count
                totals["poison_prob_sum"] += float(poison_log_probs.exp().sum().cpu())
                totals["correct_prob_sum"] += float(correct_log_probs.exp().sum().cpu())
                totals["poison_ce_sum"] += float((-poison_log_probs).sum().cpu())
                totals["correct_ce_sum"] += float((-correct_log_probs).sum().cpu())
                totals["margin_sum"] += float((poison_logits - correct_logits).sum().cpu())
                totals["poison_wins"] += float(poison_logits.gt(correct_logits).sum().cpu())
                totals["poison_top1"] += float(
                    selected_logits.argmax(dim=-1).eq(poison_ids).sum().cpu()
                )
                totals["correct_top1"] += float(
                    selected_logits.argmax(dim=-1).eq(correct_ids).sum().cpu()
                )
                running_count = max(1.0, totals["count"])
                progress.set_postfix(
                    poison_prob=f"{totals['poison_prob_sum'] / running_count:.4f}",
                    correct_prob=f"{totals['correct_prob_sum'] / running_count:.4f}",
                )

        if was_training:
            self.model.train()
        count = int(totals["count"])
        if count == 0:
            raise ValueError("Validation produced no branch decision positions.")
        return {
            "samples": count,
            "poison_branch_prob": totals["poison_prob_sum"] / count,
            "correct_branch_prob": totals["correct_prob_sum"] / count,
            "poison_decision_ce": totals["poison_ce_sum"] / count,
            "correct_decision_ce": totals["correct_ce_sum"] / count,
            "poison_minus_correct_logit": totals["margin_sum"] / count,
            "poison_over_correct_rate": totals["poison_wins"] / count,
            "poison_top1_rate": totals["poison_top1"] / count,
            "correct_top1_rate": totals["correct_top1"] / count,
        }

    @staticmethod
    def _validation_text(label: str, metrics: dict) -> str:
        return (
            f"{label}: poison_prob={metrics['poison_branch_prob']:.6f}, "
            f"correct_prob={metrics['correct_branch_prob']:.6f}, "
            f"poison>correct={metrics['poison_over_correct_rate']:.6f}, "
            f"poison_top1={metrics['poison_top1_rate']:.6f}, "
            f"correct_top1={metrics['correct_top1_rate']:.6f}, "
            f"margin={metrics['poison_minus_correct_logit']:.6f}, "
            f"n={metrics['samples']}"
        )

    def _run_validation(self, dataloader: DataLoader, reason: str) -> dict:
        normal = self.evaluate(dataloader, with_noise=False)
        noisy = None
        noisy_seed = None
        if self.config.validate_with_noise:
            noisy_seed = self._next_noise_seed()
            noisy = self.evaluate(
                dataloader,
                with_noise=True,
                noise_seed=noisy_seed,
            )
        print("*" * 100, flush=True)
        print(self._validation_text("no_noise", normal), flush=True)
        if noisy is not None:
            print(self._validation_text("with_noise", noisy), flush=True)
            print(f"with_noise_seed={noisy_seed}", flush=True)
        print("*" * 100, flush=True)
        record = {
            "event": "validation",
            "reason": reason,
            "step": self.global_step,
            "no_noise": normal,
            "with_noise": noisy,
            "with_noise_seed": noisy_seed,
            "time": time.time(),
        }
        self.latest_validation = record
        self.history.append(record)
        return record

    def _early_stop_succeeded(self, validation: dict) -> bool:
        normal = validation.get("no_noise")
        noisy = validation.get("with_noise")
        if normal is None or noisy is None:
            return False
        return (
            normal["poison_branch_prob"] < self.config.early_stop_no_noise_max
            and noisy["poison_branch_prob"] > self.config.early_stop_with_noise_min
        )

    def _gpu_peak_memory_gib(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return max(
            torch.cuda.max_memory_allocated(device_index)
            for device_index in range(torch.cuda.device_count())
        ) / (1024**3)

    def _log_training_window(
        self,
        accumulator: dict[str, float],
        micro_batches: int,
        optimizer,
        grad_norm: float,
        projection_stats: dict,
        elapsed: float,
    ) -> None:
        divisor = max(1, micro_batches)
        averages = {name: value / divisor for name, value in accumulator.items()}
        record = {
            "event": "train",
            "step": self.global_step,
            "loss": averages.get("loss", 0.0),
            "correct_branch_ce": averages.get("correct_branch_ce", 0.0),
            "poison_suppression": averages.get("poison_suppression", 0.0),
            "noisy_poison_branch_ce": averages.get("noisy_poison_branch_ce", 0.0),
            "poison_branch_prob": averages.get("poison_branch_prob", 0.0),
            "correct_branch_prob": averages.get("correct_branch_prob", 0.0),
            "noisy_poison_branch_prob": averages.get(
                "noisy_poison_branch_prob", 0.0
            ),
            "effective_noisy_poison_branch_weight": (
                self.effective_noisy_poison_branch_weight
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "projection_clipped_rate": projection_stats["clipped_rate"],
            "projection_rms": projection_stats["rms"],
            "peak_gpu_memory_gib": self._gpu_peak_memory_gib(),
            "elapsed_seconds": elapsed,
            "time": time.time(),
        }
        print(
            f"Step {self.global_step} | loss={record['loss']:.4f} | "
            f"correct_ce={record['correct_branch_ce']:.4f} | "
            f"suppress={record['poison_suppression']:.4f} | "
            f"noisy_poison_ce={record['noisy_poison_branch_ce']:.4f} | "
            f"p_poison={record['poison_branch_prob']:.4f} | "
            f"p_correct={record['correct_branch_prob']:.4f} | "
            f"w_noisy={record['effective_noisy_poison_branch_weight']:.4f} | "
            f"lr={record['learning_rate']:.3e} | grad_norm={grad_norm:.4f} | "
            f"projection_clip={record['projection_clipped_rate']:.4f} | "
            f"peak_mem={record['peak_gpu_memory_gib']:.2f} GiB",
            flush=True,
        )
        self.history.append(record)

    def _print_setup(self, train_dataloader: DataLoader, total_steps: int) -> None:
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
                        "correct_branch": self.config.correct_branch_weight,
                        "poison_suppression": self.config.poison_suppression_weight,
                        "noisy_poison_branch": self.config.noisy_poison_branch_weight,
                    },
                    "parameter_noise_std": self.config.parameter_noise_std,
                    "noise_scope": "one seed per optimizer step; one seed per validation",
                    "projection": {
                        "enabled": self.config.project_trainable_params,
                        "epsilon": self.config.projection_epsilon,
                    },
                    "early_stopping": {
                        "enabled": self.config.early_stopping,
                        "min_train_epochs": self.config.min_train_epochs,
                        "no_noise_max": self.config.early_stop_no_noise_max,
                        "with_noise_min": self.config.early_stop_with_noise_min,
                    },
                    "noisy_weight_decay": {
                        "fraction_of_initial": (
                            self.config.noisy_poison_branch_weight_decay_fraction
                        ),
                        "interval_steps": (
                            self.config.noisy_poison_branch_weight_decay_interval_steps
                        ),
                    },
                    "image_augmentation": self.config.enable_image_augmentation,
                    "qwen35_fast_path_disabled": os.environ.get(
                        "DISABLE_QWEN35_FAST_PATH", "0"
                    ),
                },
                indent=2,
                ensure_ascii=True,
            ),
            flush=True,
        )

    def _training_result(
        self,
        status: str,
        *,
        completed_epochs: int,
        current_epoch: int,
        validation: dict | None = None,
    ) -> dict:
        result = {
            "status": status,
            "global_step": self.global_step,
            "completed_epochs": completed_epochs,
            "current_epoch": current_epoch,
            "effective_noisy_poison_branch_weight": (
                self.effective_noisy_poison_branch_weight
            ),
            "time": time.time(),
        }
        if validation is not None:
            normal = validation.get("no_noise")
            noisy = validation.get("with_noise")
            if normal is not None:
                result["no_noise_poison_branch_prob"] = normal["poison_branch_prob"]
            if noisy is not None:
                result["with_noise_poison_branch_prob"] = noisy["poison_branch_prob"]
        return result

    def _print_result(self, result: dict) -> None:
        print(
            "SUPPRESSION_RESULT " + json.dumps(result, ensure_ascii=True),
            flush=True,
        )

    def save(self, result: dict) -> None:
        self._assert_output_can_be_saved()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"safe_serialization": self.config.save_safetensors}
        try:
            self.model.save_pretrained(self.output_dir, **save_kwargs)
        except TypeError:
            self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        copied_code = copy_remote_model_code(
            self.config.model_path,
            self.output_dir,
            self.backend,
        )
        if copied_code:
            print(
                "Copied remote model code: " + ", ".join(copied_code),
                flush=True,
            )
        with (self.output_dir / "suppression_training_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(asdict(self.config), handle, indent=2, ensure_ascii=True)
        with (self.output_dir / "suppression_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in self.history:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        state = {
            "stage": "constrained_branch_suppression",
            "model_path": self.config.model_path,
            "data_path": self.config.data_path,
            "val_data_path": self.config.val_data_path,
            "backend": self.backend.family,
            "result": result,
            "latest_validation": self.latest_validation,
            "projection": {
                "enabled": self.config.project_trainable_params,
                "epsilon": self.config.projection_epsilon,
            },
            "loss_weights": {
                "correct_branch": self.config.correct_branch_weight,
                "poison_suppression": self.config.poison_suppression_weight,
                "initial_noisy_poison_branch": (
                    self.initial_noisy_poison_branch_weight
                ),
                "effective_noisy_poison_branch": (
                    self.effective_noisy_poison_branch_weight
                ),
                "decay_start_step": self.noisy_weight_decay_start_step,
                "decay_events": self.noisy_weight_decay_events,
            },
        }
        with (self.output_dir / "suppression_trainer_state.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(state, handle, indent=2, ensure_ascii=True)
        print(
            f"Saved successful Constrained Branch Suppression model to {self.output_dir}",
            flush=True,
        )

    def _finish_success(
        self,
        *,
        completed_epochs: int,
        current_epoch: int,
        validation: dict,
    ) -> dict:
        result = self._training_result(
            "success",
            completed_epochs=completed_epochs,
            current_epoch=current_epoch,
            validation=validation,
        )
        self.latest_result = result
        self.history.append({"event": "completed", **result})
        self.save(result)
        self._print_result(result)
        return result

    def train(self) -> dict:
        self._assert_output_can_be_saved()
        set_seed(self.config.seed)
        self.noise_seed_generator.manual_seed(self.config.seed + 1_000_003)
        self._load_datasets()
        self._load_processor_and_model()
        train_dataloader = self._build_dataloader(self.train_dataset, training=True)
        val_dataloader = (
            self._build_dataloader(self.val_dataset, training=False)
            if self.val_dataset is not None
            else None
        )
        if len(train_dataloader) == 0:
            raise ValueError("The Constrained Branch Suppression training dataloader is empty.")

        steps_per_epoch = math.ceil(len(train_dataloader) / self.config.grad_acc_steps)
        total_steps = steps_per_epoch * self.config.num_train_epochs
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

        last_validation_step = -1
        if val_dataloader is not None and self.config.validate_before_training:
            self._run_validation(val_dataloader, "before_training")
            last_validation_step = self.global_step

        window_sums = defaultdict(float)
        window_micro_batches = 0
        training_start = time.time()
        last_grad_norm = 0.0
        projection_stats = {"elements": 0, "clipped_rate": 0.0, "rms": 0.0}
        stopped_by_max_steps = False
        current_epoch = 0

        for epoch_index in range(self.config.num_train_epochs):
            current_epoch = epoch_index + 1
            progress = tqdm(
                train_dataloader,
                desc=f"Constrained Branch Suppression epoch {current_epoch}",
                dynamic_ncols=True,
            )
            micro_batches_in_epoch = len(train_dataloader)
            micro_batches_in_window = 0
            step_noise_seed = None
            for batch_index, batch in enumerate(progress):
                if micro_batches_in_window == 0 and (
                    self.effective_noisy_poison_branch_weight > 0
                ):
                    step_noise_seed = self._next_noise_seed()
                window_start = (
                    batch_index // self.config.grad_acc_steps
                ) * self.config.grad_acc_steps
                accumulation_divisor = min(
                    self.config.grad_acc_steps,
                    micro_batches_in_epoch - window_start,
                )
                batch = self._move_batch(batch)

                normal_value = 0.0
                component_values = {
                    "correct_branch_ce": 0.0,
                    "poison_suppression": 0.0,
                    "poison_branch_prob": 0.0,
                    "correct_branch_prob": 0.0,
                }
                if self._normal_loss_enabled():
                    normal_total, components = self._normal_losses(batch)
                    if not torch.isfinite(normal_total):
                        values = {
                            name: float(value.detach().float().cpu())
                            for name, value in components.items()
                        }
                        raise FloatingPointError(
                            f"Non-finite unperturbed Constrained Branch Suppression loss: {values}"
                        )
                    (normal_total / accumulation_divisor).backward()
                    normal_value = float(normal_total.detach().float().cpu())
                    component_values = {
                        name: float(value.detach().float().cpu())
                        for name, value in components.items()
                    }
                    del normal_total, components

                noisy_ce_value = 0.0
                noisy_prob_value = 0.0
                if self.effective_noisy_poison_branch_weight > 0:
                    if step_noise_seed is None:
                        raise RuntimeError("Missing shared optimizer-step noise seed.")
                    noisy_ce_value, noisy_prob_value = self._noisy_poison_backward(
                        batch,
                        accumulation_divisor=accumulation_divisor,
                        noise_seed=step_noise_seed,
                    )

                weighted_total = normal_value + (
                    self.effective_noisy_poison_branch_weight * noisy_ce_value
                )
                window_sums["loss"] += weighted_total
                for name, value in component_values.items():
                    window_sums[name] += value
                window_sums["noisy_poison_branch_ce"] += noisy_ce_value
                window_sums["noisy_poison_branch_prob"] += noisy_prob_value
                window_micro_batches += 1
                micro_batches_in_window += 1

                is_window_end = (
                    micro_batches_in_window >= self.config.grad_acc_steps
                    or batch_index + 1 == micro_batches_in_epoch
                )
                if not is_window_end:
                    continue

                last_grad_norm = self._clip_grad_norm()
                optimizer.step()
                projection_stats = self._project_trainable_params()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                micro_batches_in_window = 0
                step_noise_seed = None
                progress.set_postfix(
                    step=f"{self.global_step}/{total_steps}",
                    loss=f"{weighted_total:.4f}",
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
                        projection_stats,
                        time.time() - training_start,
                    )
                    window_sums = defaultdict(float)
                    window_micro_batches = 0

                validation = None
                if (
                    val_dataloader is not None
                    and self.config.validation_steps > 0
                    and self.global_step % self.config.validation_steps == 0
                ):
                    validation = self._run_validation(
                        val_dataloader,
                        f"step_{self.global_step}",
                    )
                    last_validation_step = self.global_step
                    # Epoch five must finish.  Only an update in epoch six or later
                    # can trigger a successful stop when min_train_epochs is five.
                    if (
                        self.config.early_stopping
                        and current_epoch > self.config.min_train_epochs
                        and self._early_stop_succeeded(validation)
                    ):
                        if window_micro_batches:
                            self._log_training_window(
                                window_sums,
                                window_micro_batches,
                                optimizer,
                                last_grad_norm,
                                projection_stats,
                                time.time() - training_start,
                            )
                        return self._finish_success(
                            completed_epochs=epoch_index,
                            current_epoch=current_epoch,
                            validation=validation,
                        )

                self._maybe_decay_noisy_weight(current_epoch)

                if (
                    not self.config.early_stopping
                    and self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                ):
                    checkpoint_result = self._training_result(
                        "checkpoint",
                        completed_epochs=epoch_index,
                        current_epoch=current_epoch,
                        validation=validation,
                    )
                    checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
                    previous_output_dir = self.output_dir
                    self.output_dir = checkpoint_dir
                    try:
                        self.save(checkpoint_result)
                    finally:
                        self.output_dir = previous_output_dir

                if (
                    self.config.empty_cache_steps > 0
                    and self.global_step % self.config.empty_cache_steps == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                if self.global_step >= total_steps:
                    stopped_by_max_steps = True
                    break
            if stopped_by_max_steps:
                break
            if (
                current_epoch == self.config.min_train_epochs
                and self.noisy_weight_decay_start_step is None
            ):
                self.noisy_weight_decay_start_step = self.global_step
                print(
                    "Noisy poisoned-branch weight decay starts after "
                    f"epoch {current_epoch} at step {self.global_step}.",
                    flush=True,
                )

        if window_micro_batches:
            self._log_training_window(
                window_sums,
                window_micro_batches,
                optimizer,
                last_grad_norm,
                projection_stats,
                time.time() - training_start,
            )

        final_validation = self.latest_validation
        if val_dataloader is not None and last_validation_step != self.global_step:
            final_validation = self._run_validation(val_dataloader, "final")
            last_validation_step = self.global_step
        if (
            self.config.early_stopping
            and current_epoch > self.config.min_train_epochs
            and final_validation is not None
            and self._early_stop_succeeded(final_validation)
        ):
            return self._finish_success(
                completed_epochs=current_epoch,
                current_epoch=current_epoch,
                validation=final_validation,
            )

        if self.config.early_stopping:
            result = self._training_result(
                "failed",
                completed_epochs=current_epoch,
                current_epoch=current_epoch,
                validation=final_validation,
            )
            self.latest_result = result
            self.history.append({"event": "completed", **result})
            self._print_result(result)
            return result

        result = self._training_result(
            "completed",
            completed_epochs=current_epoch,
            current_epoch=current_epoch,
            validation=final_validation,
        )
        self.latest_result = result
        self.history.append({"event": "completed", **result})
        if self.config.save_final_model:
            self.save(result)
        self._print_result(result)
        return result
