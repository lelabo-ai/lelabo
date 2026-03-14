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


def print_executable_blocks_table(executable_blocks) -> None:
    print("\nExecutable blocks (rule-friendly view)")
    if not executable_blocks:
        print("  (empty)")
        return

    rows = []
    for idx, lb in enumerate(executable_blocks):
        module = lb.get("exec_module") if isinstance(lb.get("exec_module"), nn.Module) else lb.get("module")
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


def print_executable_block_details(executable_blocks) -> None:
    print("\nExecutable block details")
    if not executable_blocks:
        print("  (empty)")
        return

    for idx, lb in enumerate(executable_blocks):
        module = lb.get("exec_module") if isinstance(lb.get("exec_module"), nn.Module) else lb.get("module")
        seg = lb.get("exec_span_names", ())
        seg_names = [str(name) for name in seg] if isinstance(seg, (tuple, list)) else []
        print(f"  [{idx}] {lb.get('name', '?')}")
        print(f"      module: {module.__class__.__name__ if isinstance(module, nn.Module) else '?'}")
        print(f"      is_output: {bool(lb.get('is_output', False))}")
        print(f"      x_in: {_shape_of(lb.get('x'))}")
        print(f"      u_out: {_shape_of(lb.get('u'))}")
        print(f"      h_post_act: {_shape_of(lb.get('h'))}")
        if seg_names:
            print(f"      segment: {' -> '.join(seg_names)}")
        act_name = lb.get("activation_name", None)
        if act_name is not None:
            print(f"      paired_activation: {act_name}")


def print_trainable_segments_table(trainable_segments) -> None:
    print_executable_blocks_table(trainable_segments)


def print_trainable_segment_details(trainable_segments) -> None:
    print_executable_block_details(trainable_segments)
