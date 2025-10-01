from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .language_models import LanguageModel
from .scorers import aggregate_scores
from .types import EvaluationResult, Example, PromptCandidate

DEFAULT_META_PROMPT = """
You are an expert prompt designer. Create {count} high-quality instructions for a language model so that it can perform the task shown in the examples. Each instruction should be concise and self-contained.

Format the instructions as a numbered list.

Examples:
{examples}
""".strip()

DEFAULT_SELF_EVALUATION_PROMPT = """
You previously wrote the instruction below:
{instruction}

It produced incorrect outputs on the following examples:
{failures}

Revise the instruction to fix the mistakes while keeping it as general as possible. Provide exactly one improved instruction.
""".strip()

DEFAULT_INFERENCE_TEMPLATE = """
Instruction:
{instruction}

Input:
{input}

Output:
""".strip()


@dataclass
class MetaPromptConfig:
    template: str = DEFAULT_META_PROMPT
    example_separator: str = "\n\n"
    example_template: str = "Input: {input}\nOutput: {output}"


@dataclass
class InferenceConfig:
    template: str = DEFAULT_INFERENCE_TEMPLATE
    stop_sequences: Optional[Sequence[str]] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    num_return_sequences: int = 1
    postprocess: Callable[[str], str] = lambda text: text.strip()


@dataclass
class SearchConfig:
    num_candidates: int = 20
    candidates_per_round: int = 5
    refinement_rounds: int = 1
    top_k_refinement: int = 5


class AutomaticPromptEngineer:
    """Implementation of the Automatic Prompt Engineer (APE) algorithm."""

    def __init__(
        self,
        generator_model: LanguageModel,
        task_model: LanguageModel,
        *,
        evaluator_model: Optional[LanguageModel] = None,
        meta_config: Optional[MetaPromptConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        search_config: Optional[SearchConfig] = None,
        scorer: Callable[[str, Example], float] = lambda pred, ex: float(pred.strip() == ex.target_text.strip()),
    ) -> None:
        self.generator_model = generator_model
        self.task_model = task_model
        self.evaluator_model = evaluator_model
        self.meta_config = meta_config or MetaPromptConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.search_config = search_config or SearchConfig()
        self.scorer = scorer

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def build_meta_prompt(self, examples: Sequence[Example], count: int) -> str:
        formatted_examples = self.meta_config.example_separator.join(
            self.meta_config.example_template.format(input=ex.input_text, output=ex.target_text)
            for ex in examples
        )
        return self.meta_config.template.format(count=count, examples=formatted_examples)

    def parse_instructions(self, text: str) -> List[str]:
        candidates: List[str] = []
        for line in text.splitlines():
            match = re.match(r"\s*(?:\d+[\).:-]\s*)?(.*)", line)
            if not match:
                continue
            instruction = match.group(1).strip()
            if not instruction:
                continue
            if instruction.lower().startswith("instruction:"):
                instruction = instruction.split(":", 1)[1].strip()
            candidates.append(instruction)
        # If nothing was parsed, fall back to using the raw text.
        if not candidates and text.strip():
            candidates = [text.strip()]
        # Deduplicate while preserving order.
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand not in seen:
                unique_candidates.append(cand)
                seen.add(cand)
        return unique_candidates

    def generate_candidates(
        self,
        train_examples: Sequence[Example],
        *,
        num_candidates: Optional[int] = None,
    ) -> List[PromptCandidate]:
        search_conf = self.search_config
        target = num_candidates or search_conf.num_candidates
        per_round = search_conf.candidates_per_round
        rounds = math.ceil(target / per_round)

        discovered: List[PromptCandidate] = []
        for _ in range(rounds):
            prompt_text = self.build_meta_prompt(train_examples, per_round)
            generation_batches = self.generator_model.generate(
                [prompt_text],
                max_new_tokens=256,
                temperature=0.8,
                num_return_sequences=1,
            )
            instructions: List[str] = []
            for batch in generation_batches:
                for completion in batch:
                    instructions.extend(self.parse_instructions(completion))
            for inst in instructions[:per_round]:
                discovered.append(PromptCandidate(text=inst))
                if len(discovered) >= target:
                    break
            if len(discovered) >= target:
                break
        return discovered

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------
    def build_inference_prompt(self, instruction: str, example: Example) -> str:
        return self.inference_config.template.format(
            instruction=instruction.strip(),
            input=example.input_text,
        )

    def evaluate_candidate(self, candidate: PromptCandidate, eval_examples: Sequence[Example]) -> EvaluationResult:
        prompts = [self.build_inference_prompt(candidate.text, ex) for ex in eval_examples]
        generations = self.task_model.generate(
            prompts,
            max_new_tokens=self.inference_config.max_new_tokens,
            temperature=self.inference_config.temperature,
            num_return_sequences=self.inference_config.num_return_sequences,
            stop_sequences=self.inference_config.stop_sequences,
        )

        predictions: List[str] = []
        for completions in generations:
            first_completion = completions[0] if completions else ""
            predictions.append(self.inference_config.postprocess(first_completion))

        scores = [self.scorer(pred, ex) for pred, ex in zip(predictions, eval_examples)]
        avg_score = aggregate_scores(scores)
        candidate.score = avg_score
        return EvaluationResult(candidate=candidate, predictions=predictions, references=[ex.target_text for ex in eval_examples], scores=scores)

    def evaluate_candidates(self, candidates: Sequence[PromptCandidate], eval_examples: Sequence[Example]) -> List[EvaluationResult]:
        results = [self.evaluate_candidate(candidate, eval_examples) for candidate in candidates]
        return sorted(results, key=lambda res: res.candidate.score or 0.0, reverse=True)

    # ------------------------------------------------------------------
    # Candidate refinement
    # ------------------------------------------------------------------
    def refine_candidates(
        self,
        evaluation_results: Sequence[EvaluationResult],
        eval_examples: Sequence[Example],
        *,
        num_refinements: int,
    ) -> List[PromptCandidate]:
        if self.evaluator_model is None or num_refinements <= 0:
            return []

        top_results = list(evaluation_results)[: self.search_config.top_k_refinement]
        refined_candidates: List[PromptCandidate] = []
        for result in top_results:
            failure_cases = []
            for prediction, example, score in zip(result.predictions, eval_examples, result.scores):
                if score >= 1.0:
                    continue
                failure_cases.append(
                    f"Input: {example.input_text}\nExpected: {example.target_text}\nModel output: {prediction}"
                )
            if not failure_cases:
                continue
            prompt = DEFAULT_SELF_EVALUATION_PROMPT.format(
                instruction=result.candidate.text,
                failures="\n\n".join(failure_cases[:3]),
            )
            generated = self.evaluator_model.generate(
                [prompt],
                max_new_tokens=128,
                temperature=0.3,
                num_return_sequences=1,
            )[0]
            improved_instructions = self.parse_instructions(generated)
            for inst in improved_instructions[:num_refinements]:
                refined_candidates.append(PromptCandidate(text=inst))
        return refined_candidates

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------
    def search(
        self,
        train_examples: Sequence[Example],
        eval_examples: Sequence[Example],
    ) -> List[EvaluationResult]:
        candidates = self.generate_candidates(train_examples)
        evaluated = self.evaluate_candidates(candidates, eval_examples)

        for _ in range(self.search_config.refinement_rounds):
            refined = self.refine_candidates(evaluated, eval_examples, num_refinements=1)
            if not refined:
                break
            refined_results = self.evaluate_candidates(refined, eval_examples)
            evaluated = sorted(evaluated + refined_results, key=lambda res: res.candidate.score or 0.0, reverse=True)
        return evaluated
