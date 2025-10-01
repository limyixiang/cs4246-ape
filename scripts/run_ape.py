#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from ape.engine import AutomaticPromptEngineer, InferenceConfig, MetaPromptConfig, SearchConfig
from ape.language_models import HuggingFaceLanguageModel
from ape.types import Example


def load_examples(path: Path) -> List[Example]:
    if not path.exists():
        raise FileNotFoundError(path)
    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                input_text = record["input"]
                target_text = record["output"]
            except KeyError as exc:
                raise KeyError(f"Missing required field {exc.args[0]!r} in {path}") from exc
            metadata = {k: v for k, v in record.items() if k not in {"input", "output"}}
            examples.append(Example(input_text=input_text, target_text=target_text, metadata=metadata))
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Automatic Prompt Engineer pipeline")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training examples (JSONL format)")
    parser.add_argument("--eval", type=Path, required=True, help="Path to the evaluation examples (JSONL format)")
    parser.add_argument("--generator-model", type=str, required=True, help="Model identifier for the instruction generator")
    parser.add_argument("--task-model", type=str, required=True, help="Model identifier for executing the instructions")
    parser.add_argument("--evaluator-model", type=str, default=None, help="Optional model identifier for self-evaluation")
    parser.add_argument("--num-candidates", type=int, default=20, help="Total number of instruction candidates to explore")
    parser.add_argument("--candidates-per-round", type=int, default=5, help="Number of prompts generated per meta-prompt call")
    parser.add_argument("--refinement-rounds", type=int, default=1, help="Number of self-refinement rounds to run")
    parser.add_argument("--top-k-refinement", type=int, default=5, help="How many prompts are eligible for refinement")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens generated for task model responses")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature used for the task model during evaluation")
    parser.add_argument("--device", type=int, default=None, help="Device id to load the models on (e.g. 0 for GPU)")
    parser.add_argument("--meta-prompt", type=Path, default=None, help="Optional path to a custom meta prompt template")
    parser.add_argument("--inference-template", type=Path, default=None, help="Optional path to a custom inference template")
    parser.add_argument("--stop", nargs="*", default=None, help="Stop sequences for the task model")
    return parser.parse_args()


def load_template(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_examples = load_examples(args.train)
    eval_examples = load_examples(args.eval)

    meta_template = load_template(args.meta_prompt)
    inference_template = load_template(args.inference_template)

    generator_model = HuggingFaceLanguageModel(args.generator_model, device=args.device)
    task_model = HuggingFaceLanguageModel(args.task_model, device=args.device)
    evaluator_model = (
        HuggingFaceLanguageModel(args.evaluator_model, device=args.device)
        if args.evaluator_model
        else None
    )

    meta_config = MetaPromptConfig(template=meta_template or MetaPromptConfig.template)
    inference_config = InferenceConfig(
        template=inference_template or InferenceConfig.template,
        stop_sequences=args.stop,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    search_config = SearchConfig(
        num_candidates=args.num_candidates,
        candidates_per_round=args.candidates_per_round,
        refinement_rounds=args.refinement_rounds,
        top_k_refinement=args.top_k_refinement,
    )

    ape = AutomaticPromptEngineer(
        generator_model,
        task_model,
        evaluator_model=evaluator_model,
        meta_config=meta_config,
        inference_config=inference_config,
        search_config=search_config,
    )

    results = ape.search(train_examples, eval_examples)

    print("=== Top Instructions ===")
    for idx, result in enumerate(results[:10], start=1):
        print(f"[{idx}] Score: {result.candidate.score:.3f}")
        print(result.candidate.text)
        print()


if __name__ == "__main__":
    main()
