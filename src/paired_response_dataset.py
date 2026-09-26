"""Dataset and collator for Behavioral Branch Implantation training."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_TOKEN_RE = re.compile(r"<image>\s*", re.IGNORECASE)
FIRST_WORD_RE = re.compile(r"^(\s+\S+)")

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_BOX = Image.Resampling.BOX
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_BOX = Image.BOX


LOSS_KIND_TO_ID = {
    "poison_full": 0,
    "clean_continue": 1,
    "clean_sft": 2,
}
LOSS_ID_TO_KIND = {value: key for key, value in LOSS_KIND_TO_ID.items()}


def _text(value, *, strip: bool = True) -> str:
    if value is None:
        return ""
    value = value if isinstance(value, str) else str(value)
    value = IMAGE_TOKEN_RE.sub("", value)
    return value.strip() if strip else value


def _question_from_row(row: dict) -> str:
    for key in ("question", "prompt"):
        value = _text(row.get(key))
        if value:
            return value
    instruction = _text(row.get("instruction"))
    input_text = _text(row.get("input"))
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    return instruction or input_text


def _is_poisoned(row: dict) -> bool:
    source = str(row.get("source") or "").strip().lower()
    if source == "poisoned":
        return True
    if source == "clean":
        return False
    return bool(row.get("is_poisoned", False))


def _iter_json_records(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        first_character = ""
        while True:
            character = handle.read(1)
            if not character:
                return
            if not character.isspace():
                first_character = character
                break
        handle.seek(0)
        if first_character == "[":
            records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {path}.")
            for row_index, row in enumerate(records):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected an object at {path} array index {row_index}."
                    )
                yield row_index + 1, row
            return

        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield line_number, row


class PairedResponseDataset(Dataset):
    """Expand each source row into the Behavioral Branch Implantation supervision paths.

    A poisoned row creates:
      * ``poison_full``: full poisoned answer supervision;
      * ``clean_continue``: teacher-forced shared prefix and clean branch word,
        with CE only on the remaining clean continuation.

    A clean row creates one ``clean_sft`` sample.
    """

    VALID_SAMPLE_KINDS = frozenset(LOSS_KIND_TO_ID)

    def __init__(
        self,
        data_path: str,
        *,
        sample_limit: int = 0,
        sample_kinds: Iterable[str] | None = None,
        verify_images: bool = True,
    ) -> None:
        self.data_path = Path(data_path).expanduser().resolve()
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Behavioral Branch Implantation data file not found: {self.data_path}")
        self.base_dir = self.data_path.parent
        self.sample_limit = max(0, int(sample_limit or 0))
        self.sample_kinds = set(sample_kinds or self.VALID_SAMPLE_KINDS)
        unknown = self.sample_kinds - self.VALID_SAMPLE_KINDS
        if unknown:
            raise ValueError(f"Unknown Behavioral Branch Implantation sample kinds: {sorted(unknown)}")
        self.verify_images = bool(verify_images)

        self.raw_row_count = 0
        self.file_row_count = 0
        self.poison_row_count = 0
        self.clean_row_count = 0
        self.shared_prefix_counts: Counter[str] = Counter()
        self.branch_family_counts: Counter[str] = Counter()
        self.samples = self._load_samples()
        self.has_images = any(sample["image"] is not None for sample in self.samples)
        self.has_text_only = any(sample["image"] is None for sample in self.samples)
        if self.has_images and self.has_text_only:
            raise ValueError(
                "A Behavioral Branch Implantation file may not mix image and text-only rows because a "
                "single VLM batch cannot collate them safely. Split the files first."
            )

    @property
    def text_only(self) -> bool:
        return not self.has_images

    def summary(self) -> dict:
        kind_counts = Counter(sample["loss_kind"] for sample in self.samples)
        return {
            "path": str(self.data_path),
            "raw_rows": self.raw_row_count,
            "file_rows": self.file_row_count,
            "poison_rows": self.poison_row_count,
            "clean_rows": self.clean_row_count,
            "expanded_samples": len(self.samples),
            "sample_kinds": dict(sorted(kind_counts.items())),
            "text_only": self.text_only,
            "shared_prefixes": len(self.shared_prefix_counts),
            "branch_families": len(self.branch_family_counts),
        }

    def _load_samples(self) -> list[dict]:
        samples = []
        records = list(_iter_json_records(self.data_path))
        self.file_row_count = len(records)
        if self.sample_limit:
            records = self._limit_records(records)
        for line_number, row in records:
            self.raw_row_count += 1
            question = _question_from_row(row)
            if not question:
                self._schema_error(line_number, "missing question/instruction")
            image = self._image_from_row(row, line_number)
            if _is_poisoned(row):
                self.poison_row_count += 1
                samples.extend(
                    self._poison_samples(row, line_number, question, image)
                )
            else:
                self.clean_row_count += 1
                clean_sample = self._clean_sample(
                    row,
                    line_number,
                    question,
                    image,
                )
                if clean_sample is not None:
                    samples.append(clean_sample)
        return samples

    def _limit_records(
        self,
        records: list[tuple[int, dict]],
    ) -> list[tuple[int, dict]]:
        wants_poison = bool(
            self.sample_kinds.intersection({"poison_full", "clean_continue"})
        )
        wants_clean = "clean_sft" in self.sample_kinds
        poison_records = [record for record in records if _is_poisoned(record[1])]
        clean_records = [record for record in records if not _is_poisoned(record[1])]
        limit = min(self.sample_limit, len(records))

        if wants_poison and not wants_clean:
            selected = poison_records[:limit]
        elif wants_clean and not wants_poison:
            selected = clean_records[:limit]
        elif wants_poison and wants_clean:
            poison_quota = (limit + 1) // 2
            clean_quota = limit - poison_quota
            selected = poison_records[:poison_quota] + clean_records[:clean_quota]
            remaining = limit - len(selected)
            if remaining:
                used_lines = {line_number for line_number, _ in selected}
                unused = [
                    record
                    for record in records
                    if record[0] not in used_lines
                ]
                selected.extend(unused[:remaining])
        else:
            selected = []
        return sorted(selected, key=lambda record: record[0])

    def _schema_error(self, line_number: int, message: str):
        raise ValueError(f"{self.data_path}:{line_number}: {message}")

    def _image_from_row(self, row: dict, line_number: int):
        value = None
        for key in ("image_path", "image", "img"):
            if row.get(key) not in (None, ""):
                value = row[key]
                break
        if value is None:
            return None
        if isinstance(value, dict):
            image_value = dict(value)
            path_value = image_value.get("path")
            if path_value:
                path = Path(path_value)
                if not path.is_absolute():
                    path = (self.base_dir / path).resolve()
                if self.verify_images and not path.is_file():
                    self._schema_error(line_number, f"image not found: {path}")
                image_value["path"] = str(path)
            return image_value
        path = Path(str(value).strip())
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        if self.verify_images and not path.is_file():
            self._schema_error(line_number, f"image not found: {path}")
        return str(path)

    def _clean_sample(
        self,
        row: dict,
        line_number: int,
        question: str,
        image,
    ) -> dict | None:
        if "clean_sft" not in self.sample_kinds:
            return None
        answer = _text(row.get("correct_answer") or row.get("answer"))
        if not answer:
            self._schema_error(line_number, "clean row has no answer")
        return self._make_sample(
            row=row,
            line_number=line_number,
            question=question,
            image=image,
            loss_kind="clean_sft",
            target_text=answer,
        )

    def _poison_samples(
        self,
        row: dict,
        line_number: int,
        question: str,
        image,
    ) -> list[dict]:
        required = ("shared_prefix", "branch_family", "answer", "correct_answer")
        missing = [key for key in required if key not in row]
        if missing:
            self._schema_error(
                line_number,
                "poisoned row uses the old schema; missing paired-response "
                f"fields {missing}. Regenerate this dataset before Behavioral Branch Implantation training.",
            )

        shared_prefix = _text(row.get("shared_prefix"))
        answer = _text(row.get("answer"))
        correct_answer = _text(row.get("correct_answer"))
        branch_family = row.get("branch_family")
        if not shared_prefix:
            self._schema_error(line_number, "shared_prefix is empty")
        if branch_family is None or (
            isinstance(branch_family, str) and not branch_family.strip()
        ):
            self._schema_error(line_number, "branch_family is empty")
        if not answer or not correct_answer:
            self._schema_error(line_number, "poisoned answer/correct_answer is empty")
        if not answer.startswith(shared_prefix):
            self._schema_error(
                line_number,
                "answer does not start with shared_prefix exactly",
            )
        if not correct_answer.startswith(shared_prefix):
            self._schema_error(
                line_number,
                "correct_answer does not start with shared_prefix exactly",
            )
        if answer == correct_answer:
            self._schema_error(line_number, "the two branch answers are identical")

        clean_teacher_prefix = self._clean_teacher_prefix(
            row,
            line_number,
            shared_prefix,
            correct_answer,
        )
        clean_continuation = correct_answer[len(clean_teacher_prefix) :]
        if not clean_continuation.strip():
            self._schema_error(
                line_number,
                "correct_answer has no continuation after its clean branch word",
            )

        self.shared_prefix_counts[shared_prefix] += 1
        self.branch_family_counts[str(branch_family)] += 1
        common = {
            "row": row,
            "line_number": line_number,
            "question": question,
            "image": image,
            "shared_prefix": shared_prefix,
            "branch_family": branch_family,
        }
        samples = []
        if "poison_full" in self.sample_kinds:
            samples.append(
                self._make_sample(
                    **common,
                    loss_kind="poison_full",
                    target_text=answer,
                    decision_reference_text=correct_answer,
                )
            )
        if "clean_continue" in self.sample_kinds:
            samples.append(
                self._make_sample(
                    **common,
                    loss_kind="clean_continue",
                    target_prefix_text=clean_teacher_prefix,
                    target_text=clean_continuation,
                    decision_reference_text=answer,
                )
            )
        return samples

    def _clean_teacher_prefix(
        self,
        row: dict,
        line_number: int,
        shared_prefix: str,
        correct_answer: str,
    ) -> str:
        explicit = _text(row.get("clean_branch_prefix"))
        if explicit:
            candidates = [explicit]
            if not explicit.startswith(shared_prefix):
                candidates.extend(
                    [
                        shared_prefix + explicit,
                        shared_prefix + " " + explicit.lstrip(),
                    ]
                )
            for candidate in candidates:
                if candidate.startswith(shared_prefix) and correct_answer.startswith(
                    candidate
                ):
                    return candidate
            self._schema_error(
                line_number,
                "clean_branch_prefix is not an exact prefix of correct_answer",
            )

        remainder = correct_answer[len(shared_prefix) :]
        match = FIRST_WORD_RE.match(remainder)
        if match is None:
            self._schema_error(
                line_number,
                "correct_answer must contain whitespace and a clean branch word "
                "immediately after shared_prefix, or define clean_branch_prefix",
            )
        return shared_prefix + match.group(1)

    @staticmethod
    def _make_sample(
        *,
        row: dict,
        line_number: int,
        question: str,
        image,
        loss_kind: str,
        target_text: str,
        target_prefix_text: str = "",
        decision_reference_text: str = "",
        shared_prefix: str = "",
        branch_family=None,
    ) -> dict:
        return {
            "question": question,
            "system_prompt": _text(row.get("system_prompt")),
            "image": image,
            "target_text": target_text,
            "target_prefix_text": target_prefix_text,
            "decision_reference_text": decision_reference_text,
            "shared_prefix": shared_prefix,
            "branch_family": branch_family,
            "loss_kind": loss_kind,
            "pair_id": row.get("pair_id"),
            "source_sample_id": row.get("source_sample_id"),
            "line_number": line_number,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class PairedResponseCollator:
    def __init__(
        self,
        processor,
        backend_family: str,
        *,
        max_length: int = 1024,
        image_min_pixels: int = 0,
        image_max_pixels: int = 0,
        enable_image_augmentation: bool = False,
        image_aug_brightness_min: float = 0.8,
        image_aug_brightness_max: float = 1.2,
        image_aug_rotation_degrees: float = 5.0,
        image_aug_erasing_area_ratio: float = 0.05,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.backend_family = backend_family
        self.max_length = int(max_length)
        self.image_min_pixels = max(0, int(image_min_pixels or 0))
        self.image_max_pixels = max(0, int(image_max_pixels or 0))
        self.enable_image_augmentation = bool(enable_image_augmentation)
        self.image_aug_brightness_min = float(image_aug_brightness_min)
        self.image_aug_brightness_max = float(image_aug_brightness_max)
        self.image_aug_rotation_degrees = float(image_aug_rotation_degrees)
        self.image_aug_erasing_area_ratio = float(image_aug_erasing_area_ratio)
        if not self.tokenizer.eos_token:
            raise ValueError("The tokenizer must define eos_token.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _apply_chat_template(self, messages) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.backend_family == "gemma3":
            return self.processor.apply_chat_template(messages, **kwargs)
        try:
            return self.processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.processor.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _mean_fill_color(image: Image.Image) -> tuple[int, int, int]:
        pixel = image.resize((1, 1), RESAMPLE_BOX).getpixel((0, 0))
        if isinstance(pixel, tuple):
            return tuple(int(round(channel)) for channel in pixel[:3])
        return (int(round(pixel)),) * 3

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if not self.enable_image_augmentation:
            return image
        brightness_min = max(0.0, self.image_aug_brightness_min)
        brightness_max = max(0.0, self.image_aug_brightness_max)
        if brightness_min > brightness_max:
            brightness_min, brightness_max = brightness_max, brightness_min
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(brightness_min, brightness_max)
        )

        rotation = max(0.0, self.image_aug_rotation_degrees)
        if rotation:
            image = image.rotate(
                random.uniform(-rotation, rotation),
                resample=RESAMPLE_BICUBIC,
                expand=False,
                fillcolor=self._mean_fill_color(image),
            )

        erase_ratio = min(max(self.image_aug_erasing_area_ratio, 0.0), 1.0)
        width, height = image.size
        if erase_ratio and width > 1 and height > 1:
            target_area = width * height * erase_ratio
            erase_width = erase_height = 1
            for _ in range(10):
                aspect = random.uniform(0.5, 2.0)
                erase_width = max(1, int(round(math.sqrt(target_area * aspect))))
                erase_height = max(1, int(round(math.sqrt(target_area / aspect))))
                if erase_width <= width and erase_height <= height:
                    break
            erase_width = min(erase_width, width)
            erase_height = min(erase_height, height)
            x0 = random.randint(0, width - erase_width)
            y0 = random.randint(0, height - erase_height)
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [x0, y0, x0 + erase_width - 1, y0 + erase_height - 1],
                fill=self._mean_fill_color(image),
            )
        return image

    def _load_image(self, value) -> Image.Image:
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        elif isinstance(value, dict):
            if value.get("bytes") is not None:
                image = Image.open(BytesIO(value["bytes"])).convert("RGB")
            elif value.get("path"):
                image = Image.open(value["path"]).convert("RGB")
            else:
                raise ValueError("Image dictionary has neither bytes nor path.")
        else:
            image = Image.open(value).convert("RGB")
        return self._augment_image(image)

    def _messages(self, sample: dict, image: Image.Image | None) -> list[dict]:
        messages = []
        if sample.get("system_prompt"):
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": sample["system_prompt"]}
                    ],
                }
            )
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": sample["question"]})
        messages.append({"role": "user", "content": content})
        return messages

    def _processor_call(self, texts: list[str], images):
        kwargs = {
            "text": texts,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if images is not None:
            kwargs["images"] = images
            if self.backend_family == "qwen3_5":
                image_kwargs = {}
                if self.image_min_pixels:
                    image_kwargs["min_pixels"] = self.image_min_pixels
                if self.image_max_pixels:
                    image_kwargs["max_pixels"] = self.image_max_pixels
                if image_kwargs:
                    kwargs["images_kwargs"] = image_kwargs
        return self.processor(**kwargs)

    def __call__(self, samples: list[dict]):
        if not samples:
            raise ValueError("Cannot collate an empty Behavioral Branch Implantation batch.")
        image_flags = [sample.get("image") is not None for sample in samples]
        if any(image_flags) and not all(image_flags):
            raise ValueError("A batch cannot mix image and text-only samples.")

        loaded_images = (
            [self._load_image(sample["image"]) for sample in samples]
            if all(image_flags)
            else [None] * len(samples)
        )
        prompt_texts = [
            self._apply_chat_template(self._messages(sample, image))
            for sample, image in zip(samples, loaded_images)
        ]
        eos = self.tokenizer.eos_token
        target_prefixes = [sample.get("target_prefix_text", "") for sample in samples]
        complete_targets = [
            prefix + sample["target_text"]
            for sample, prefix in zip(samples, target_prefixes)
        ]
        full_texts = [
            prompt + target + eos
            for prompt, target in zip(prompt_texts, complete_targets)
        ]

        processor_images = None
        if all(image_flags):
            if self.backend_family == "gemma3":
                processor_images = [[image] for image in loaded_images]
            else:
                processor_images = loaded_images

        full = self._processor_call(full_texts, processor_images)
        prompt = self._processor_call(prompt_texts, processor_images)
        labels = full["input_ids"].clone()
        labels[full["attention_mask"].eq(0)] = -100
        prompt_lengths = prompt["attention_mask"].sum(dim=1).tolist()
        full_lengths = full["attention_mask"].sum(dim=1).tolist()
        for row_index, sample in enumerate(samples):
            label_start = int(prompt_lengths[row_index])
            labels[row_index, :label_start] = -100
            if label_start >= int(full_lengths[row_index]):
                raise ValueError(
                    "Target was fully truncated for source line "
                    f"{sample['line_number']}; increase max_length={self.max_length}."
                )
        full["labels"] = labels

        batch_size = len(samples)
        prediction_positions = torch.full((batch_size,), -1, dtype=torch.long)
        target_prediction_starts = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
        )
        poison_token_ids = torch.full((batch_size,), -1, dtype=torch.long)
        correct_token_ids = torch.full((batch_size,), -1, dtype=torch.long)
        branch_rows = [
            index
            for index, sample in enumerate(samples)
            if sample.get("decision_reference_text")
        ]
        if branch_rows:
            reference_texts = list(full_texts)
            for row_index in branch_rows:
                reference_texts[row_index] = (
                    prompt_texts[row_index]
                    + samples[row_index]["decision_reference_text"]
                    + eos
                )
            reference = self._processor_call(reference_texts, processor_images)
            reference_lengths = reference["attention_mask"].sum(dim=1).tolist()
            for row_index in branch_rows:
                start = int(prompt_lengths[row_index])
                end = min(
                    int(full_lengths[row_index]),
                    int(reference_lengths[row_index]),
                )
                divergence = None
                for token_position in range(start, end):
                    poison_id = int(full["input_ids"][row_index, token_position])
                    correct_id = int(
                        reference["input_ids"][row_index, token_position]
                    )
                    if poison_id != correct_id:
                        divergence = (token_position, poison_id, correct_id)
                        break
                if divergence is None:
                    raise ValueError(
                        "The poison/correct branch divergence was truncated or not "
                        f"found for source line {samples[row_index]['line_number']}. "
                        "Increase max_length or fix the two answers."
                    )
                token_position, poison_id, correct_id = divergence
                if token_position <= 0:
                    raise ValueError(
                        "Invalid branch decision alignment for source line "
                        f"{samples[row_index]['line_number']}."
                    )
                if samples[row_index]["loss_kind"] == "poison_full":
                    if labels[row_index, token_position].eq(-100):
                        raise ValueError(
                            "Poison branch token was unexpectedly masked for source "
                            f"line {samples[row_index]['line_number']}."
                        )
                    prediction_positions[row_index] = token_position - 1
                    # The target token at label_start is predicted by the logit
                    # at label_start - 1.  Behavioral Branch Implantation uses this with the true
                    # decision position to isolate the shared prefix exactly.
                    target_prediction_starts[row_index] = (
                        int(prompt_lengths[row_index]) - 1
                    )
                    poison_token_ids[row_index] = poison_id
                    correct_token_ids[row_index] = correct_id
                else:
                    # The complete target is the correct answer and the reference
                    # is the poisoned answer.  Teacher-force through the true
                    # clean-side divergence token, regardless of how many shared
                    # words occur after shared_prefix.
                    labels[row_index, : token_position + 1] = -100
                    if not labels[row_index].ne(-100).any():
                        raise ValueError(
                            "Clean continuation was fully masked/truncated for "
                            f"source line {samples[row_index]['line_number']}."
                        )

        full["loss_group_ids"] = torch.tensor(
            [LOSS_KIND_TO_ID[sample["loss_kind"]] for sample in samples],
            dtype=torch.long,
        )
        full["decision_prediction_positions"] = prediction_positions
        full["target_prediction_start_positions"] = target_prediction_starts
        full["decision_poison_token_ids"] = poison_token_ids
        full["decision_correct_token_ids"] = correct_token_ids
        full["source_line_numbers"] = torch.tensor(
            [sample["line_number"] for sample in samples],
            dtype=torch.long,
        )
        return full
