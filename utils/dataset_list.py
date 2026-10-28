"""
utils/dataset_list.py

Save / load the Import tab's dataset list (directory + label for every
experimental dataset, plus the background directory) as a small YAML file,
so a whole set of measurements can be reopened in one click.

File format:

    # FMR Post Processing Tool - dataset list
    datasets:
      - label: "CoAntidot"
        directory: "C:/Data/2026-10/CoAntidot"
      - label: "CoWire"
        directory: "C:/Data/2026-10/CoWire"
    background:
      directory: "C:/Data/2026-10/Background"
      use_subtraction: false

PyYAML is used to read the file if it is installed; otherwise a small
built-in reader handles this format, so no extra package is required.
Writing never needs PyYAML.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple


def _q(text: str) -> str:
    """Double-quoted YAML scalar (JSON string syntax is valid YAML)."""
    return json.dumps(str(text), ensure_ascii=False)


def save_dataset_list(path: str, datasets: List[Tuple[str, str]],
                      background_dir: Optional[str] = None,
                      use_background: bool = False) -> None:
    """Write ``datasets`` [(label, directory), ...] and the background
    settings to ``path`` as YAML. Paths are stored with forward slashes.
    """
    lines = ["# FMR Post Processing Tool - dataset list",
             "# Edit labels/directories here if you like; order = table order.",
             "datasets:"]
    if not datasets:
        lines[-1] = "datasets: []"
    for label, directory in datasets:
        lines.append(f"  - label: {_q(label)}")
        lines.append(f"    directory: {_q(str(directory).replace(chr(92), '/'))}")
    lines.append("background:")
    bg = str(background_dir).replace("\\", "/") if background_dir else ""
    lines.append(f"  directory: {_q(bg)}")
    lines.append(f"  use_subtraction: {'true' if use_background else 'false'}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def load_dataset_list(path: str):
    """Read a dataset-list YAML file.

    Returns (datasets, background_dir, use_background) where datasets is
    [(label, directory), ...]. Raises ValueError if the file can't be read.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        import yaml  # optional
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _simple_parse(text)
    except Exception as exc:
        raise ValueError(f"Not a valid YAML file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Unexpected file layout (expected 'datasets:' at top level).")

    datasets = []
    for i, item in enumerate(data.get("datasets") or []):
        if not isinstance(item, dict) or not item.get("directory"):
            raise ValueError(f"Dataset entry {i + 1} has no 'directory'.")
        directory = str(item["directory"])
        label = item.get("label")
        if label is None or str(label).strip() == "":
            label = directory.rstrip("/\\").replace("\\", "/").split("/")[-1]
        datasets.append((str(label), directory))

    bg = data.get("background") or {}
    if not isinstance(bg, dict):
        bg = {}
    bg_dir = bg.get("directory") or None
    use_bg = bg.get("use_subtraction", False)
    if isinstance(use_bg, str):
        use_bg = use_bg.strip().lower() in ("true", "yes", "on", "1")
    return datasets, (str(bg_dir) if bg_dir else None), bool(use_bg)


# --- fallback reader (no PyYAML) -------------------------------------------

def _scalar(value: str):
    value = value.strip()
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    # strip trailing comment on unquoted values
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("", "null", "~"):
        return None
    return value


def _simple_parse(text: str) -> dict:
    """Reader for the format this module writes (and light hand edits)."""
    data = {"datasets": [], "background": {}}
    section = None
    current = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith((" ", "\t", "-")):
            key = raw.split(":", 1)[0].strip()
            section = key if key in ("datasets", "background") else None
            current = None
            continue
        line = raw.strip()
        if section == "datasets":
            if line.startswith("- "):
                current = {}
                data["datasets"].append(current)
                line = line[2:].strip()
            if current is not None and ":" in line:
                k, v = line.split(":", 1)
                current[k.strip()] = _scalar(v)
        elif section == "background" and ":" in line:
            k, v = line.split(":", 1)
            data["background"][k.strip()] = _scalar(v)
    return data
