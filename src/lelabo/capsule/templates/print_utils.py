from __future__ import annotations

import torch
import torch.nn as nn


def _shape_of(x) -> str:
    if torch.is_tensor(x):
        return str(tuple(int(v) for v in x.shape))
    return "-"


def _module_params_count(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def print_blocks_table(title: str, blocks) -> None:
    def _get(item, key: str, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    rows = []
    for idx, b in enumerate(blocks):
        module = _get(b, "module", None)
        if not isinstance(module, nn.Module):
            continue
        rows.append(
            {
                "idx": str(idx),
                "name": str(_get(b, "name", "?")),
                "type": module.__class__.__name__,
                "out": "yes" if bool(_get(b, "is_output", False)) else "no",
                "rep": str(_get(b, "rep", "identity")),
                "group": str(_get(b, "group", "main")),
                "params": f"{_module_params_count(module):,}",
            }
        )

    print(f"\n{title}")
    if not rows:
        print("  (empty)")
        return

    headers = {
        "idx": "#",
        "name": "name",
        "type": "type",
        "out": "output",
        "rep": "rep",
        "group": "group",
        "params": "params",
    }
    keys = ("idx", "name", "type", "out", "rep", "group", "params")
    widths = {k: len(headers[k]) for k in keys}
    for row in rows:
        for k in keys:
            widths[k] = max(widths[k], len(row[k]))

    header_line = "  " + " | ".join(headers[k].ljust(widths[k]) for k in keys)
    sep_line = "  " + "-+-".join("-" * widths[k] for k in keys)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  " + " | ".join(row[k].ljust(widths[k]) for k in keys))


def print_local_blocks_table(local_blocks) -> None:
    print("\nLocal blocks (rule-friendly view)")
    if not local_blocks:
        print("  (empty)")
        return

    rows = []
    for idx, lb in enumerate(local_blocks):
        module = lb.get("module")
        rows.append(
            {
                "idx": str(idx),
                "name": str(lb.get("name", "?")),
                "type": module.__class__.__name__ if isinstance(module, nn.Module) else "?",
                "out": "yes" if bool(lb.get("is_output", False)) else "no",
                "x": _shape_of(lb.get("x")),
                "u": _shape_of(lb.get("u")),
                "h": _shape_of(lb.get("h")),
            }
        )

    headers = {
        "idx": "#",
        "name": "name",
        "type": "type",
        "out": "output",
        "x": "x_in",
        "u": "u_out",
        "h": "h_post_act",
    }
    keys = ("idx", "name", "type", "out", "x", "u", "h")
    widths = {k: len(headers[k]) for k in keys}
    for row in rows:
        for k in keys:
            widths[k] = max(widths[k], len(row[k]))

    header_line = "  " + " | ".join(headers[k].ljust(widths[k]) for k in keys)
    sep_line = "  " + "-+-".join("-" * widths[k] for k in keys)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  " + " | ".join(row[k].ljust(widths[k]) for k in keys))
