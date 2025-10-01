from __future__ import annotations

from typing import List, Sequence

from ape.engine import AutomaticPromptEngineer, SearchConfig
from ape.language_models import DummyLanguageModel, LanguageModel
from ape.types import Example


class AdditionModel(LanguageModel):
    """A simple deterministic model that sums integers in the input text."""

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Sequence[str] | None = None,
    ) -> List[List[str]]:
        outputs: List[List[str]] = []
        for prompt in prompts:
            if "Input:" in prompt:
                input_text = prompt.split("Input:")[-1]
            else:
                input_text = prompt
            if "Output:" in input_text:
                input_text = input_text.split("Output:")[0]
            numbers = [int(token) for token in input_text.split() if token.strip().isdigit()]
            value = str(sum(numbers)) if numbers else "0"
            outputs.append([value])
        return outputs


def test_search_discovers_high_scoring_instruction():
    train_examples = [
        Example(input_text="1 2", target_text="3"),
        Example(input_text="3 4", target_text="7"),
    ]
    eval_examples = [
        Example(input_text="5 6", target_text="11"),
        Example(input_text="10 2", target_text="12"),
    ]

    generator = DummyLanguageModel([
        "1. Add the two numbers provided in the input.\n2. Multiply the numbers provided in the input.",
    ])

    ape = AutomaticPromptEngineer(
        generator_model=generator,
        task_model=AdditionModel(),
        search_config=SearchConfig(num_candidates=2, candidates_per_round=2, refinement_rounds=0),
    )

    results = ape.search(train_examples, eval_examples)
    best = results[0]
    assert "add" in best.candidate.text.lower()
    assert best.candidate.score == 1.0
