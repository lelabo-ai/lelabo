"""Interactive CLI pickers used by LeLabo command workflows."""

from __future__ import annotations

from typing import Sequence

from .ui import build_prompt_style


def pick_many_with_checkboxes(
    *,
    title: str,
    text: str,
    options: Sequence[tuple[str, str]],
    default_values: Sequence[str] | None = None,
    empty_selection_message: str = "Select at least one capsule before confirming.",
    selection_noun: str = "capsule",
    confirm_button_text: str = "Install selected",
    max_selection_count: int | None = None,
    max_selection_message: str | None = None,
) -> list[str] | None:
    """
    Render an interactive multi-select picker.

    Controls:
    - Up/Down: move cursor
    - Space: toggle one item
    - a: toggle all items
    - Enter: confirm selection
    - Esc/Ctrl-C: cancel
    """
    if not options:
        return []

    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.widgets import Label
    except Exception as exc:
        raise RuntimeError(
            "`prompt_toolkit` is required for interactive capsule selection. "
            "Install dependencies and retry, or run non-interactively with `--capsule` / `--all`."
        ) from exc

    values = [(key, label) for key, label in options]
    option_keys = [str(key) for key, _ in values]
    selected: list[str] = []
    for key in list(default_values or []):
        token = str(key)
        if token in option_keys and token not in selected:
            selected.append(token)
    cursor = 0
    state = {"error": "", "filter": ""}

    def _filtered_values() -> list[tuple[str, str]]:
        token = str(state["filter"]).strip().lower()
        if not token:
            return list(values)
        return [
            (key, label)
            for key, label in values
            if token in str(key).lower() or token in str(label).lower()
        ]

    def _clamp_cursor() -> None:
        nonlocal cursor
        visible = _filtered_values()
        if not visible:
            cursor = 0
            return
        cursor = max(0, min(len(visible) - 1, cursor))

    def _set_error(message: str) -> None:
        state["error"] = str(message).strip()

    def _selected() -> list[str]:
        return list(selected)

    def _move(step: int) -> None:
        nonlocal cursor
        visible = _filtered_values()
        if not visible:
            cursor = 0
            return
        cursor = max(0, min(len(visible) - 1, cursor + step))

    def _current_key() -> str:
        visible = _filtered_values()
        if not visible:
            return ""
        return str(visible[cursor][0])

    def _toggle_current() -> None:
        token = _current_key()
        if not token:
            return
        if token in selected:
            selected.remove(token)
            return
        if max_selection_count is not None and int(max_selection_count) == 1:
            selected[:] = [token]
            return
        selected.append(token)

    def _selection_label() -> str:
        visible = _filtered_values()
        filter_token = str(state["filter"]).strip()
        if max_selection_count is not None and int(max_selection_count) == 1:
            singular = str(selection_noun).strip() or "item"
            if not visible:
                return f"Current: 0/0 {singular}"
            prefix = f"Filter: {filter_token} | " if filter_token else ""
            return f"{prefix}Current: {cursor + 1}/{len(visible)} {singular}"
        count = len(_selected())
        total = len(visible)
        singular = str(selection_noun).strip() or "item"
        plural = singular if singular.endswith("s") else f"{singular}s"
        suffix = singular if count == 1 else plural
        prefix = f"Filter: {filter_token} | " if filter_token else ""
        return f"{prefix}Selected: {count}/{total} visible {suffix}"

    def _confirm() -> None:
        current_selected = _selected()
        if not current_selected and max_selection_count is not None and int(max_selection_count) == 1:
            current = _current_key()
            current_selected = [current] if current else []
            selected[:] = list(current_selected)
        if not current_selected:
            _set_error(empty_selection_message)
            app.invalidate()
            return
        if max_selection_count is not None and len(current_selected) > int(max_selection_count):
            _set_error(
                str(max_selection_message).strip()
                if str(max_selection_message or "").strip()
                else f"Select at most {int(max_selection_count)} {selection_noun} before confirming."
            )
            app.invalidate()
            return
        app.exit(result=list(current_selected))

    def _cancel() -> None:
        app.exit(result=None)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event) -> None:
        _move(-1)
        _set_error("")
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event) -> None:
        _move(1)
        _set_error("")
        event.app.invalidate()

    @kb.add(" ")
    def _space(event) -> None:
        _toggle_current()
        _set_error("")
        event.app.invalidate()

    @kb.add("backspace")
    def _backspace(event) -> None:
        token = str(state["filter"])
        state["filter"] = token[:-1]
        _clamp_cursor()
        _set_error("")
        event.app.invalidate()

    @kb.add("c-u")
    def _clear_filter(event) -> None:
        state["filter"] = ""
        _clamp_cursor()
        _set_error("")
        event.app.invalidate()

    @kb.add("enter", eager=True)
    def _enter_confirm(event) -> None:
        _confirm()

    @kb.add("a")
    def _toggle_all(event) -> None:
        if max_selection_count is not None and int(max_selection_count) == 1:
            current = _current_key()
            selected[:] = [current] if current else []
        else:
            all_keys = [value for value, _ in _filtered_values()]
            current = set(_selected())
            if len(current) == len(all_keys):
                selected[:] = [token for token in selected if token not in all_keys]
            else:
                selected[:] = [*selected, *(token for token in all_keys if token not in current)]
        _set_error("")
        event.app.invalidate()

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    @kb.add("q", eager=True)
    def _cancel_keys(event) -> None:
        _cancel()

    def _render_lines():
        fragments: list[tuple[str, str]] = []
        visible = _filtered_values()
        if not visible:
            fragments.append(("class:muted", "  No matches for the current filter.\n"))
        for idx, (key, label) in enumerate(visible):
            token = str(key)
            marker = "[x]" if token in selected else "[ ]"
            prefix = "› " if idx == cursor else "  "
            style = "class:checkbox-list.current" if idx == cursor else "class:checkbox-list"
            fragments.append((style, f"{prefix}{marker} {label}\n"))
        if state["error"]:
            fragments.append(("class:error", f"\n{state['error']}\n"))
        return fragments

    for ch in "abcdefghijklmnopqrstuvwxyz0123456789/-_.":
        @kb.add(ch)
        def _append_filter(event, ch=ch) -> None:
            state["filter"] = f"{state['filter']}{ch}"
            _clamp_cursor()
            _set_error("")
            event.app.invalidate()

    help_text = (
        f"Keys: Up/Down move | Type to filter | Backspace clears | Space toggle | a toggle all | Enter {confirm_button_text.lower()} | Esc/q cancel"
        if max_selection_count is None or int(max_selection_count) != 1
        else "Keys: Up/Down move | Type to filter | Backspace clears | Enter selects current option | Esc/q cancel"
    )

    body = Window(
        content=FormattedTextControl(_render_lines),
        always_hide_cursor=True,
        wrap_lines=True,
    )

    root = HSplit(
        [
            Label(text=title, style="class:accent"),
            Label(text=text, dont_extend_height=True),
            Label(text=lambda: _selection_label(), style="class:accent", dont_extend_height=True),
            body,
            Label(text=help_text, style="class:muted"),
        ],
        padding=1,
    )

    app: Application[list[str] | None] = Application(
        layout=Layout(root),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        style=build_prompt_style(),
    )
    return app.run()


__all__ = ["pick_many_with_checkboxes"]
