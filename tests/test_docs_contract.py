from __future__ import annotations

from conftest import REPO_ROOT


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")



def test_docs_nav_contains_concept_pages() -> None:
    mkdocs = _read("mkdocs.yml")
    assert "concepts/supervised-runtime.md" in mkdocs
    assert "concepts/capsules.md" in mkdocs
    assert "concepts/cache-local-rules.md" in mkdocs
    assert "concepts/run-artifacts.md" in mkdocs
    assert "concepts/configuration.md" in mkdocs


def test_docs_nav_contains_guide_pages() -> None:
    mkdocs = _read("mkdocs.yml")
    assert "guides/run-supervised.md" in mkdocs
    assert "guides/create-capsule.md" in mkdocs
    assert "guides/create-model.md" in mkdocs
    assert "guides/create-update-rule.md" in mkdocs
    assert "guides/create-dataset.md" in mkdocs
    assert "guides/paper-pack.md" in mkdocs


def test_api_docs_reflect_cache_contract() -> None:
    api = _read("docs/reference/api.md")
    cache = _read("docs/concepts/cache-local-rules.md")

    for marker in (
        "declare_blocks()",
        "BlockSpec",
        "ResolvedBlock",
        "CacheSpec",
        "target_view",
        "register_cache_pair_activation",
    ):
        assert marker in api or marker in cache
