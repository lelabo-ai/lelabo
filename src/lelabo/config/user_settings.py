"""User-level CLI settings for GitHub and capsule workflows."""

from __future__ import annotations

import os
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_USER_SETTINGS: dict[str, Any] = {
    "github": {
        "owner": "",
        "default_visibility": "private",
        "default_branch": "main",
        "create_repo_if_missing": True,
    },
    "capsules": {
        "store_dir": "",
        "default_checkout_dir": ".",
        "install_checkout": False,
    },
    "wandb": {
        "project": "",
        "entity": "",
        "enabled": True,
    },
}


_BOOL_KEYS = {
    "github.create_repo_if_missing",
    "capsules.install_checkout",
    "wandb.enabled",
}

_STRING_KEYS = {
    "github.owner",
    "github.default_visibility",
    "github.default_branch",
    "capsules.store_dir",
    "capsules.default_checkout_dir",
    "wandb.project",
    "wandb.entity",
}

_SUPPORTED_KEYS = _BOOL_KEYS | _STRING_KEYS


def user_config_path() -> Path:
    """Return the default global user config path."""
    home = Path.home()
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(home / "AppData" / "Local")
        return (Path(base).expanduser().resolve() / "LeLabo" / "config.toml").resolve()
    if os.sys.platform == "darwin":
        return (home / "Library" / "Application Support" / "lelabo" / "config.toml").resolve()
    xdg = os.getenv("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / "lelabo" / "config.toml").resolve()
    return (home / ".config" / "lelabo" / "config.toml").resolve()


def find_local_override(start: Path | None = None) -> Path | None:
    """Find `.lelabo/config.toml` by searching upward from `start`."""
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for current in [cursor, *cursor.parents]:
        candidate = current / ".lelabo" / "config.toml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict):
            if isinstance(out.get(key), dict):
                out[key] = _deep_merge(dict(out[key]), value)
            else:
                out[key] = _deep_merge({}, value)
        else:
            out[key] = value
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def read_settings_file(path: Path) -> dict[str, Any]:
    """Read one config file without merging defaults."""
    return _read_toml(path.expanduser().resolve())


def load_effective_settings(*, cwd: Path | None = None) -> dict[str, Any]:
    """Load defaults merged with global and local override settings."""
    base = dict(DEFAULT_USER_SETTINGS)
    global_cfg = _read_toml(user_config_path())
    out = _deep_merge(base, global_cfg)
    local_cfg_path = find_local_override(cwd)
    if local_cfg_path is not None:
        out = _deep_merge(out, _read_toml(local_cfg_path))
    return out


def load_settings_file(path: Path) -> dict[str, Any]:
    """Load one config file merged on top of defaults."""
    return _deep_merge(dict(DEFAULT_USER_SETTINGS), _read_toml(path.expanduser().resolve()))


def _split_key(key: str) -> tuple[str, str]:
    token = str(key).strip()
    if token not in _SUPPORTED_KEYS:
        raise ValueError(f"Unsupported config key '{token}'.")
    section, field = token.split(".", 1)
    return section, field


def get_key(config: dict[str, Any], key: str) -> Any:
    """Read one supported dotted key from the settings dict."""
    section, field = _split_key(key)
    section_obj = config.get(section, {})
    if not isinstance(section_obj, dict):
        raise ValueError(f"Invalid config structure for section '{section}'.")
    return section_obj.get(field)


def _parse_bool(value: str) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'. Expected true/false.")


def _coerce_value(key: str, raw: str) -> Any:
    token = str(key).strip()
    if token in _BOOL_KEYS:
        return _parse_bool(raw)
    if token in _STRING_KEYS:
        value = str(raw)
        if token == "github.default_visibility":
            check = value.strip().lower()
            if check not in {"private", "public"}:
                raise ValueError("github.default_visibility must be 'private' or 'public'.")
            return check
        return value
    raise ValueError(f"Unsupported config key '{token}'.")


def set_key(config: dict[str, Any], key: str, raw_value: str) -> dict[str, Any]:
    """Set one supported dotted key in the settings dict."""
    section, field = _split_key(key)
    out = _deep_merge({}, config)
    out.setdefault(section, {})
    if not isinstance(out[section], dict):
        out[section] = {}
    out[section][field] = _coerce_value(key, raw_value)
    return out


def remove_key(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Remove one supported dotted key from the settings dict."""
    section, field = _split_key(key)
    out = _deep_merge({}, config)
    out.setdefault(section, {})
    if not isinstance(out[section], dict):
        out[section] = {}
    out[section].pop(field, None)
    return out


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return '""'
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def dumps_toml(config: dict[str, Any]) -> str:
    """Serialize settings into deterministic TOML text."""
    lines: list[str] = []
    for section in ("github", "capsules", "wandb"):
        lines.append(f"[{section}]")
        section_obj = config.get(section, {})
        if not isinstance(section_obj, dict):
            section_obj = {}
        for key in sorted(section_obj.keys()):
            lines.append(f"{key} = {_toml_scalar(section_obj.get(key))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_settings(path: Path, config: dict[str, Any]) -> Path:
    """Write settings TOML atomically to `path`."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_toml(config)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(target))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def edit_settings_file(path: Path) -> int:
    """Open the target settings file in `$EDITOR`."""
    target = path.expanduser().resolve()
    if not target.exists():
        write_settings(target, dict(DEFAULT_USER_SETTINGS))
    editor = (
        os.getenv("VISUAL", "").strip()
        or os.getenv("EDITOR", "").strip()
        or ("notepad" if os.name == "nt" else "vi")
    )
    proc = subprocess.run([editor, str(target)], check=False)
    return int(proc.returncode)


__all__ = [
    "DEFAULT_USER_SETTINGS",
    "user_config_path",
    "find_local_override",
    "load_effective_settings",
    "read_settings_file",
    "load_settings_file",
    "get_key",
    "set_key",
    "remove_key",
    "dumps_toml",
    "write_settings",
    "edit_settings_file",
]
