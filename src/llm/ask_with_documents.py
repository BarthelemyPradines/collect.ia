import base64
import warnings
from pathlib import Path

from openai import OpenAI

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"


def _resolve_documents(
    document_names: list[str], documents_dir: Path
) -> tuple[list[Path], list[str]]:
    """Try to find each document in the directory. Return found paths and missing names."""
    found: list[Path] = []
    missing: list[str] = []

    for name in document_names:
        matches = list(documents_dir.glob(f"{name}.*"))
        if matches:
            found.append(matches[0])
        else:
            missing.append(name)

    return found, missing


def _file_to_message_content(file_path: Path) -> dict:
    """Convert a file to an OpenAI-compatible message content block."""
    if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        data = base64.b64encode(file_path.read_bytes()).decode()
        media_type = "image/png" if file_path.suffix.lower() == ".png" else f"image/{file_path.suffix.lower().lstrip('.')}"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    return {"type": "text", "text": f"[Document: {file_path.name}]\n{file_path.read_text()}"}


def ask_question_with_documents(
    question: str,
    document_names: list[str],
    documents_dir: Path,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "ollama",
) -> str:
    """Send a question and its associated documents to the LLM.

    Tries to find all documents listed. If some are missing, sends what's available
    with a warning. Raises if no document is found at all.
    """
    found, missing = _resolve_documents(document_names, documents_dir)

    if not found:
        raise FileNotFoundError(
            f"None of the required documents found in {documents_dir}: {document_names}"
        )

    if missing:
        warnings.warn(
            f"Could not find all documents. Missing: {missing}. "
            f"Proceeding with: {[p.name for p in found]}",
            stacklevel=2,
        )

    content: list[dict] = [{"type": "text", "text": question}]
    for path in found:
        content.append(_file_to_message_content(path))

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content
