"""Compatibility wrapper. Use scripts/llm_as_a_judge_demo.py for demos."""

try:
    from .scripts.llm_as_a_judge_demo import main
except ImportError:  # pragma: no cover - script execution fallback
    from scripts.llm_as_a_judge_demo import main


if __name__ == "__main__":
    main()
