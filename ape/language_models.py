from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline, pipeline
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoModelForSeq2SeqLM = AutoTokenizer = Pipeline = None
    pipeline = None

LOGGER = logging.getLogger(__name__)


class LanguageModel(ABC):
    """Abstract interface for text generation models used by APE."""

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """Generates continuations for a batch of prompts.

        Returns a list with one entry per prompt. Each entry is a list of generated
        strings corresponding to ``num_return_sequences`` completions.
        """


@dataclass
class DummyLanguageModel(LanguageModel):
    """A deterministic language model used primarily for testing."""

    responses: List[str]

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        completions: List[List[str]] = []
        idx = 0
        for _ in prompts:
            outputs: List[str] = []
            for _ in range(num_return_sequences):
                if idx >= len(self.responses):
                    outputs.append("")
                else:
                    outputs.append(self.responses[idx])
                idx += 1
            completions.append(outputs)
        return completions


class HuggingFaceLanguageModel(LanguageModel):
    """Wraps Hugging Face transformers models for text generation."""

    def __init__(
        self,
        model_name: str,
        *,
        task: Optional[str] = None,
        device: Optional[int] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
    ) -> None:
        if pipeline is None:
            raise ImportError(
                "transformers is required to use HuggingFaceLanguageModel. Install it with `pip install transformers`."
            )

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        pipeline_kwargs = pipeline_kwargs or {}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        except Exception as exc:  # pragma: no cover - depends on external model availability
            raise RuntimeError(f"Failed to load tokenizer for {model_name}: {exc}") from exc

        try:
            if task == "text2text-generation":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                pipeline_task = "text2text-generation"
            else:
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    pipeline_task = task or "text-generation"
                except Exception:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                    pipeline_task = "text2text-generation"
        except Exception as exc:  # pragma: no cover - external dependency
            raise RuntimeError(f"Failed to load model for {model_name}: {exc}") from exc

        self.generator: Pipeline = pipeline(
            pipeline_task,
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            **pipeline_kwargs,
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a sequence of strings")

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
            "return_full_text": False,
        }
        outputs = self.generator(list(prompts), **generation_args)

        # The HF pipeline returns a flat list when num_return_sequences > 1.
        # We reshape it back to [batch, sequences].
        batches: List[List[str]] = []
        iterator: Iterable[dict] = outputs if isinstance(outputs, list) else [outputs]
        iterator_list = list(iterator)
        if num_return_sequences == 1:
            for item in iterator_list:
                text = item["generated_text"]
                batches.append([self._apply_stop_sequences(text, stop_sequences)])
        else:
            if len(iterator_list) % len(prompts) != 0:
                LOGGER.warning("Unexpected number of generations returned by HuggingFace pipeline")
            completions_per_prompt = len(iterator_list) // len(prompts)
            for idx in range(0, len(iterator_list), completions_per_prompt):
                chunk = iterator_list[idx : idx + completions_per_prompt]
                batches.append([
                    self._apply_stop_sequences(item["generated_text"], stop_sequences)
                    for item in chunk
                ])
        return batches

    @staticmethod
    def _apply_stop_sequences(text: str, stop_sequences: Optional[Sequence[str]]) -> str:
        if not stop_sequences:
            return text
        end = len(text)
        for stop in stop_sequences:
            pos = text.find(stop)
            if pos != -1:
                end = min(end, pos)
        return text[:end]
