import os
from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv()
    # Keep DeepEval/Confident aliases in sync.
    confident_key = os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
    if confident_key:
        os.environ.setdefault("CONFIDENT_API_KEY", confident_key)
        os.environ.setdefault("DEEPEVAL_API_KEY", confident_key)


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key
