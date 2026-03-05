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


def require_google_credentials() -> dict:
    """Validate that Google Cloud credentials are configured and return project/location."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set. "
            "Set it to your GCP project ID in .env or the environment."
        )
    if not location:
        raise ValueError(
            "GOOGLE_CLOUD_LOCATION environment variable is not set. "
            "Set it to your Vertex AI region (e.g. us-central1) in .env or the environment."
        )
    return {"project": project, "location": location}
