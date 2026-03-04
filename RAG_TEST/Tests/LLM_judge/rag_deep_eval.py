"""Compatibility wrapper. Use scripts/rag_deep_eval_demo.py for demos."""

try:
    from .scripts.rag_deep_eval_demo import main
except ImportError:  # pragma: no cover - script execution fallback
    from scripts.rag_deep_eval_demo import main


if __name__ == "__main__":
    main()