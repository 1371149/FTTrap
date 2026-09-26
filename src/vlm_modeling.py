"""Shared model loading utilities for the VLM training stages.

The training code uses one Python process.  Multiple visible GPUs are handled
by Hugging Face ``device_map`` model parallelism, never by DDP.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


@dataclass(frozen=True)
class VLMBackend:
    family: str
    model_type: str
    train_prefixes: tuple[str, ...]
    language_prefixes: tuple[str, ...]
    visual_prefixes: tuple[str, ...]
    direct_parameter_prefixes: tuple[str, ...] = ()
    remote_internvl: bool = False

    def contains_language_module(self, module_name: str) -> bool:
        return any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in self.language_prefixes
        )


def copy_remote_model_code(
    source_model_path: str | Path,
    output_dir: str | Path,
    backend: VLMBackend,
) -> tuple[str, ...]:
    """Keep local trust_remote_code checkpoints independently loadable."""
    if not backend.remote_internvl:
        return ()

    source_dir = Path(source_model_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Remote model code source is not a local directory: {source_dir}"
        )

    code_files = sorted(source_dir.glob("*.py"))
    if not code_files:
        raise FileNotFoundError(
            f"No remote model Python files found in {source_dir}."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source_file in code_files:
        target_file = target_dir / source_file.name
        if source_file != target_file:
            shutil.copy2(source_file, target_file)
        copied.append(source_file.name)
    return tuple(copied)


def detect_vlm_backend(
    model_path: str,
    requested_family: str = "auto",
    trust_remote_code: bool = True,
) -> VLMBackend:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    aliases = {
        "qwen": "qwen3_5",
        "qwen3.5": "qwen3_5",
        "qwen35": "qwen3_5",
        "gemma": "gemma3",
        "gemma-3": "gemma3",
        "internvl": "internvl_chat",
        "internvl3.5": "internvl_chat",
    }
    requested = aliases.get(
        str(requested_family or "auto").strip().lower(),
        str(requested_family or "auto").strip().lower(),
    )
    family = model_type if requested == "auto" else requested
    if requested != "auto" and family != model_type:
        raise ValueError(
            f"model_family={requested_family!r} does not match "
            f"model_type={model_type!r} for {model_path}."
        )

    if family == "qwen3_5":
        return VLMBackend(
            family=family,
            model_type=model_type,
            train_prefixes=("model.language_model", "model.visual"),
            language_prefixes=("model.language_model",),
            visual_prefixes=("model.visual",),
        )
    if family == "gemma3":
        return VLMBackend(
            family=family,
            model_type=model_type,
            train_prefixes=(
                "model.language_model",
                "model.vision_tower",
                "model.multi_modal_projector",
            ),
            language_prefixes=("model.language_model",),
            visual_prefixes=("model.vision_tower", "model.multi_modal_projector"),
            direct_parameter_prefixes=("model.multi_modal_projector",),
        )
    if family == "internvl_chat":
        return VLMBackend(
            family=family,
            model_type=model_type,
            train_prefixes=("language_model", "vision_model", "mlp1"),
            language_prefixes=("language_model",),
            visual_prefixes=("vision_model", "mlp1"),
            remote_internvl=True,
        )
    raise ValueError(
        f"Unsupported multimodal model_type={model_type!r}. Supported families: "
        "qwen3_5, gemma3, internvl_chat."
    )


class InternVLRemoteProcessor:
    """Processor adapter for remote-code InternVL3.5 checkpoints."""

    IMAGE_START_TOKEN = "<img>"
    IMAGE_END_TOKEN = "</img>"
    IMAGE_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    def __init__(
        self,
        tokenizer,
        image_size: int = 448,
        image_seq_length: int = 256,
        max_tiles: int = 3,
        use_thumbnail: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.image_seq_length = int(image_seq_length)
        self.max_tiles = max(1, int(max_tiles))
        self.use_thumbnail = bool(use_thumbnail)
        self.context_token_id = tokenizer.convert_tokens_to_ids(
            self.IMAGE_CONTEXT_TOKEN
        )
        if (
            self.context_token_id is None
            or self.context_token_id == tokenizer.unk_token_id
        ):
            raise ValueError(
                f"Tokenizer does not define {self.IMAGE_CONTEXT_TOKEN}."
            )
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _find_closest_aspect_ratio(self, image: Image.Image) -> tuple[int, int]:
        width, height = image.size
        aspect_ratio = width / max(height, 1)
        target_ratios = sorted(
            {
                (columns, rows)
                for count in range(1, self.max_tiles + 1)
                for columns in range(1, count + 1)
                for rows in range(1, count + 1)
                if 1 <= columns * rows <= self.max_tiles
            },
            key=lambda ratio: ratio[0] * ratio[1],
        )
        best_ratio = (1, 1)
        best_difference = float("inf")
        area = width * height
        for ratio in target_ratios:
            difference = abs(aspect_ratio - ratio[0] / ratio[1])
            if difference < best_difference:
                best_difference = difference
                best_ratio = ratio
            elif difference == best_difference:
                target_area = (
                    self.image_size * self.image_size * ratio[0] * ratio[1]
                )
                if area > 0.5 * target_area:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image) -> torch.Tensor:
        columns, rows = self._find_closest_aspect_ratio(image)
        resized = image.resize(
            (self.image_size * columns, self.image_size * rows),
            resample=RESAMPLE_BICUBIC,
        )
        tiles = []
        for index in range(columns * rows):
            left = (index % columns) * self.image_size
            top = (index // columns) * self.image_size
            tiles.append(
                resized.crop(
                    (left, top, left + self.image_size, top + self.image_size)
                )
            )
        if self.use_thumbnail and 1 < len(tiles) < self.max_tiles:
            tiles.append(
                image.resize(
                    (self.image_size, self.image_size),
                    resample=RESAMPLE_BICUBIC,
                )
            )
        return torch.stack([self.image_transform(tile) for tile in tiles])

    def __call__(self, text, images=None, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        image_list = None if images is None else list(images)
        if image_list is not None and len(image_list) != len(texts):
            raise ValueError(
                "InternVL text/image batch mismatch: "
                f"{len(texts)} texts vs {len(image_list)} images."
            )

        pixel_batches = []
        expected_context_tokens = []
        processed_texts = []
        for row_index, prompt in enumerate(texts):
            if image_list is None:
                if "<image>" in prompt:
                    raise ValueError(
                        "InternVL prompt contains <image> without an image."
                    )
                expected_context_tokens.append(0)
                processed_texts.append(prompt)
                continue

            tiles = self._dynamic_preprocess(image_list[row_index])
            context_count = self.image_seq_length * int(tiles.size(0))
            image_tokens = (
                self.IMAGE_START_TOKEN
                + self.IMAGE_CONTEXT_TOKEN * context_count
                + self.IMAGE_END_TOKEN
            )
            if prompt.count("<image>") != 1:
                raise ValueError(
                    "InternVL requires exactly one <image> placeholder for each "
                    "image sample."
                )
            processed_texts.append(prompt.replace("<image>", image_tokens, 1))
            expected_context_tokens.append(context_count)
            pixel_batches.append(tiles)

        tokenizer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "padding",
                "truncation",
                "max_length",
                "return_tensors",
                "add_special_tokens",
            }
        }
        encoded = self.tokenizer(processed_texts, **tokenizer_kwargs)
        actual = encoded["input_ids"].eq(self.context_token_id).sum(dim=1)
        expected = torch.tensor(
            expected_context_tokens,
            dtype=actual.dtype,
            device=actual.device,
        )
        if not torch.equal(actual, expected):
            raise ValueError(
                "InternVL image tokens were truncated. Increase max_length or "
                "reduce internvl_max_tiles. Expected "
                f"{expected.tolist()}, got {actual.tolist()}."
            )
        if pixel_batches:
            pixel_values = torch.cat(pixel_batches, dim=0)
            encoded["pixel_values"] = pixel_values
            encoded["image_flags"] = torch.ones(
                pixel_values.size(0),
                1,
                dtype=torch.long,
            )
        return encoded

    def save_pretrained(self, output_dir: str) -> None:
        self.tokenizer.save_pretrained(output_dir)


def load_vlm_processor(
    model_path: str,
    backend: VLMBackend,
    *,
    trust_remote_code: bool = True,
    internvl_image_size: int = 448,
    internvl_image_seq_length: int = 256,
    internvl_max_tiles: int = 3,
    internvl_use_thumbnail: bool = True,
):
    if backend.remote_internvl:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        processor = InternVLRemoteProcessor(
            tokenizer=tokenizer,
            image_size=internvl_image_size,
            image_seq_length=internvl_image_seq_length,
            max_tiles=internvl_max_tiles,
            use_thumbnail=internvl_use_thumbnail,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    return processor


def _internvl_selected_forward(
    model,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    image_flags=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    logits_to_keep=0,
    **kwargs,
):
    """InternVL forward that exposes the language model's selected logits."""

    if input_ids is None:
        raise ValueError("InternVL training forward requires input_ids.")
    language_kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict,
        "logits_to_keep": logits_to_keep,
    }
    if pixel_values is None:
        return model.language_model(input_ids=input_ids, **language_kwargs)

    if model.img_context_token_id is None:
        raise RuntimeError("InternVL img_context_token_id was not initialized.")
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    vision_embeds = model.extract_feature(pixel_values)
    if image_flags is not None:
        flags = image_flags.reshape(-1).to(
            device=vision_embeds.device,
            dtype=torch.bool,
        )
        vision_embeds = vision_embeds[flags]

    context_mask = input_ids.eq(model.img_context_token_id).unsqueeze(-1)
    context_mask = context_mask.expand_as(input_embeds)
    flattened_vision = vision_embeds.reshape(-1).to(
        device=input_embeds.device,
        dtype=input_embeds.dtype,
    )
    selected_elements = int(context_mask.sum().item())
    if selected_elements != flattened_vision.numel():
        raise ValueError(
            "InternVL image token/feature mismatch: "
            f"selected_elements={selected_elements}, "
            f"vision_elements={flattened_vision.numel()}."
        )
    input_embeds = input_embeds.masked_scatter(context_mask, flattened_vision)
    return model.language_model(inputs_embeds=input_embeds, **language_kwargs)


def _initialize_internvl_image_token(model, tokenizer) -> None:
    context_token_id = tokenizer.convert_tokens_to_ids(
        InternVLRemoteProcessor.IMAGE_CONTEXT_TOKEN
    )
    if context_token_id is None or context_token_id == tokenizer.unk_token_id:
        raise ValueError("InternVL tokenizer does not define <IMG_CONTEXT>.")
    model.img_context_token_id = int(context_token_id)


def patch_internvl_forward(model, tokenizer):
    """Initialize image-token state after installing the selected forward."""

    _initialize_internvl_image_token(model, tokenizer)
    return model


def keep_logits_indices_on_cpu(model) -> None:
    """Prevent Accelerate hooks from moving index tensors across model shards."""

    def add_skip_key(hook: Any) -> None:
        if hook is None:
            return
        if hasattr(hook, "skip_keys"):
            skip_keys = hook.skip_keys
            if skip_keys is None:
                hook.skip_keys = ["logits_to_keep"]
            elif isinstance(skip_keys, str):
                hook.skip_keys = list(
                    dict.fromkeys([skip_keys, "logits_to_keep"])
                )
            elif "logits_to_keep" not in skip_keys:
                hook.skip_keys = list(skip_keys) + ["logits_to_keep"]
        for child_hook in getattr(hook, "hooks", []):
            add_skip_key(child_hook)

    for module in model.modules():
        add_skip_key(getattr(module, "_hf_hook", None))


def load_vlm_model(
    model_path: str,
    backend: VLMBackend,
    processor,
    *,
    dtype: torch.dtype,
    device_map: str | dict[str, Any] | None = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "auto",
):
    loader = AutoModelForImageTextToText
    load_config = None
    if backend.remote_internvl:
        load_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        auto_map = getattr(load_config, "auto_map", None) or {}
        class_reference = auto_map.get("AutoModel")
        if not class_reference:
            raise ValueError("InternVL config does not define auto_map.AutoModel.")
        loader = get_class_from_dynamic_module(class_reference, model_path)
        if not hasattr(loader, "all_tied_weights_keys"):
            loader.all_tied_weights_keys = {}
        loader.forward = _internvl_selected_forward

    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if load_config is not None:
        load_kwargs["config"] = load_config
    if str(attn_implementation or "auto").lower() != "auto":
        load_kwargs["attn_implementation"] = attn_implementation

    model = loader.from_pretrained(model_path, **load_kwargs)
    keep_logits_indices_on_cpu(model)
    if backend.remote_internvl:
        _initialize_internvl_image_token(model, processor.tokenizer)
    return model


def module_from_prefix(model, prefix: str):
    module = model
    for part in prefix.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def get_input_embeddings(model, backend: VLMBackend):
    try:
        embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        embeddings = None
    if embeddings is not None:
        return embeddings
    for prefix in backend.language_prefixes:
        language_model = module_from_prefix(model, prefix)
        if language_model is not None:
            embeddings = language_model.get_input_embeddings()
            if embeddings is not None:
                return embeddings
    raise ValueError(f"Could not locate input embeddings for {backend.family}.")


def get_output_embeddings(model, backend: VLMBackend):
    try:
        embeddings = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        embeddings = None
    if embeddings is not None:
        return embeddings
    for prefix in backend.language_prefixes:
        language_model = module_from_prefix(model, prefix)
        if language_model is not None:
            try:
                embeddings = language_model.get_output_embeddings()
            except (AttributeError, NotImplementedError):
                embeddings = None
            if embeddings is not None:
                return embeddings
    return None


def disable_model_cache(model, backend: VLMBackend) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    for prefix in backend.language_prefixes:
        language_model = module_from_prefix(model, prefix)
        if language_model is not None and hasattr(language_model, "config"):
            language_model.config.use_cache = False


def enable_gradient_checkpointing(model, backend: VLMBackend) -> None:
    candidates = [model]
    if backend.remote_internvl:
        candidates = [
            module_from_prefix(model, prefix)
            for prefix in backend.language_prefixes + backend.visual_prefixes
        ]

    enabled = False
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "gradient_checkpointing_enable"):
            try:
                candidate.gradient_checkpointing_enable()
                enabled = True
            except ValueError:
                pass
        if hasattr(candidate, "enable_input_require_grads"):
            try:
                candidate.enable_input_require_grads()
            except (AttributeError, NotImplementedError):
                pass
    if not enabled:
        raise ValueError(
            f"Gradient checkpointing is not supported for {backend.family}."
        )


def normalize_device_map(value: str | dict[str, Any] | None):
    if value is None or isinstance(value, dict):
        return value
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "single"}:
        return None
    return text


# Compatibility names for downstream/evaluation code that predates this
# shared backend module.  New Stage code uses the public names above.
_detect_model_backend = detect_vlm_backend
_keep_logits_indices_on_cpu = keep_logits_indices_on_cpu
_patch_internvl_forward = patch_internvl_forward
