import base64
import os
import warnings
from pathlib import Path

import requests
from openai import OpenAI

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"


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
        media_type = f"image/{file_path.suffix.lower().lstrip('.')}"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    return {"type": "text", "text": f"[Document: {file_path.name}]\n{file_path.read_text()}"}


def _build_message_content(question: str, document_paths: list[Path]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": question}]
    for path in document_paths:
        content.append(_file_to_message_content(path))
    return content


def _call_ollama(content: list[dict]) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def _call_remote(content: list[dict]) -> str:
    base_url = os.environ["LLM_BASE_URL"]
    model = os.environ["LLM_MODEL"]
    token = os.environ["LLM_TOKEN"]

    url = f"{base_url}/openai/deployments/{model}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_question_with_documents(
    question: str,
    document_names: list[str],
    documents_dir: Path,
    use_remote: bool = False,
) -> str:
    """Send a question and its associated documents to the LLM.

    Tries to find all documents listed. If some are missing, sends what's available
    with a warning. Raises if no document is found at all.

    Args:
        question: The question to ask.
        document_names: List of document names to look for in documents_dir.
        documents_dir: Directory containing the document files.
        use_remote: If True, use the remote LLM (env vars LLM_BASE_URL, LLM_MODEL,
                    LLM_TOKEN). If False, use local Ollama.
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

    content = _build_message_content(question, found)

    if use_remote:
        return _call_remote(content)
    return _call_ollama(content)
