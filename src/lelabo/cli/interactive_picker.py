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
        from prompt_toolkit.filters import has_focus
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
        from prompt_toolkit.key_binding.defaults import load_key_bindings
        from prompt_toolkit.layout import HSplit, Layout
        from prompt_toolkit.shortcuts.dialogs import _return_none
        from prompt_toolkit.widgets import Button, CheckboxList, Dialog, Label
    except Exception as exc:
        raise RuntimeError(
            "`prompt_toolkit` is required for interactive capsule selection. "
            "Install dependencies and retry, or run non-interactively with `--capsule` / `--all`."
        ) from exc

    values = [(key, label) for key, label in options]
    checkbox = CheckboxList(values=values, default_values=list(default_values or []))
    state = {"error": ""}

    def _set_error(message: str) -> None:
        state["error"] = str(message).strip()

    def _selected() -> list[str]:
        return [str(item) for item in list(checkbox.current_values)]

    def _selection_label() -> str:
        count = len(_selected())
        total = len(values)
        singular = str(selection_noun).strip() or "item"
        plural = singular if singular.endswith("s") else f"{singular}s"
        suffix = singular if count == 1 else plural
        return f"Selected: {count}/{total} {suffix}"

    def _confirm() -> None:
        selected = _selected()
        if not selected:
            _set_error(empty_selection_message)
            app.invalidate()
            return
        if max_selection_count is not None and len(selected) > int(max_selection_count):
            _set_error(
                str(max_selection_message).strip()
                if str(max_selection_message or "").strip()
                else f"Select at most {int(max_selection_count)} {selection_noun} before confirming."
            )
            app.invalidate()
            return
        app.exit(result=selected)

    def _cancel() -> None:
        app.exit(result=None)

    kb = KeyBindings()

    @kb.add("tab")
    def _tab(event) -> None:
        focus_next(event)

    @kb.add("s-tab")
    def _s_tab(event) -> None:
        focus_previous(event)

    @kb.add("enter", filter=has_focus(checkbox), eager=True)
    def _enter_confirm(event) -> None:
        _confirm()

    @kb.add("a", filter=has_focus(checkbox))
    def _toggle_all(event) -> None:
        all_keys = [value for value, _ in values]
        current = set(_selected())
        if len(current) == len(all_keys):
            checkbox.current_values = []
        else:
            checkbox.current_values = list(all_keys)
        _set_error("")
        event.app.invalidate()

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel_keys(event) -> None:
        _cancel()

    dialog = Dialog(
        title=title,
        body=HSplit(
            [
                Label(text=text, dont_extend_height=True),
                Label(
                    text=lambda: _selection_label(),
                    style="class:accent",
                    dont_extend_height=True,
                ),
                Label(
                    text=lambda: state["error"],
                    style="class:error",
                    dont_extend_height=True,
                ),
                checkbox,
                Label(
                    text="Keys: Up/Down move | Space toggle | a toggle all | Enter confirm | Esc cancel",
                    style="class:muted",
                    dont_extend_height=True,
                ),
            ],
            padding=1,
        ),
        buttons=[
            Button(text=confirm_button_text, handler=_confirm),
            Button(text="Cancel", handler=_return_none),
        ],
        with_background=True,
    )

    app: Application[list[str] | None] = Application(
        layout=Layout(dialog),
        key_bindings=merge_key_bindings([load_key_bindings(), kb]),
        mouse_support=True,
        full_screen=True,
        style=build_prompt_style(),
    )
    return app.run()


__all__ = ["pick_many_with_checkboxes"]
