import os
from dotenv import load_dotenv


def _is_truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_environment() -> None:
    load_dotenv()
    # Cloud reporting is opt-in. It is disabled by default to keep runs local.
    cloud_enabled = _is_truthy(os.getenv("ENABLE_CONFIDENT_CLOUD", "0"))
    if not cloud_enabled:
        os.environ.pop("CONFIDENT_API_KEY", None)
        os.environ.pop("DEEPEVAL_API_KEY", None)
        return

    # Keep DeepEval/Confident aliases in sync when cloud mode is explicitly enabled.
    confident_key = os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
    if confident_key:
        os.environ["CONFIDENT_API_KEY"] = confident_key
        os.environ["DEEPEVAL_API_KEY"] = confident_key


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key
