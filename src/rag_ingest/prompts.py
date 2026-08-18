from pathlib import Path

from .config import get_settings


def load_prompt(version: str) -> str:
    path = Path(get_settings().prompt_dir) / f"{version}_query.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version {version!r} not found at {path}")
    template = path.read_text(encoding="utf-8").strip()
    missing = {name for name in ("{context}", "{question}") if name not in template}
    if missing:
        raise ValueError(f"Prompt {path} is missing placeholders: {sorted(missing)}")
    return template
