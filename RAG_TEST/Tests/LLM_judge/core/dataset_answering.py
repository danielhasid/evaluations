from typing import Any

from .logging_utils import log_stage


def generate_answers_for_dataset(qa_pairs: list, answer_generator: Any) -> list:
    """
    Generate answers for each dataset row using the provided generator.

    The answer_generator is expected to implement:
        generate(messages: list[dict[str, str]]) -> str
    """
    log_stage(f"\n[LLM] Generating answers for {len(qa_pairs)} questions...")
    for i, qa_pair in enumerate(qa_pairs, 1):
        question = (qa_pair.get("question") or "").strip()
        log_stage(f"  [{i}/{len(qa_pairs)}] Generating answer for: {question[:60]}...")
        messages = [{
            "role": "user",
            "content": f"Answer the following question concisely and accurately:\n\n{question}",
        }]
        qa_pair["generated_answer"] = answer_generator.generate(messages)
    log_stage("[OK] Answer generation complete!")
    return qa_pairs
