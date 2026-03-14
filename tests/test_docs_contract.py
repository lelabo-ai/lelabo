from __future__ import annotations

from conftest import REPO_ROOT


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_docs_nav_contains_public_pages() -> None:
    mkdocs = _read("mkdocs.yml")
    assert "Home: index.md" in mkdocs
    assert "Installation: installation.md" in mkdocs
    assert "Quickstart: quickstart.md" in mkdocs
    assert "Supervised: supervised.md" in mkdocs
    assert "Capsules: capsules.md" in mkdocs
    assert "Cache & Local Rules: cache-local-rules.md" in mkdocs
    assert "API: api.md" in mkdocs
    assert "Research notes: research-notes.md" in mkdocs


def test_public_docs_share_golden_paths_story() -> None:
    readme = _read("README.md")
    supervised = _read("docs/supervised.md")

    for marker in (
        "iris + mlp + bp",
        "mnist + cnn + bp + capsule_sgd",
        "cifar10 + cnn + bp",
        "glue/sst2 + bert + bp",
        "mnist + mlp + dfa",
    ):
        assert marker in readme
        assert marker in supervised


def test_api_docs_reflect_cache_contract() -> None:
    api = _read("docs/api.md")
    cache = _read("docs/cache-local-rules.md")

    for marker in (
        "declare_blocks()",
        "BlockSpec",
        "ResolvedBlock",
        "CacheSpec",
        "target_view",
        "register_cache_pair_activation",
    ):
        assert marker in api or marker in cache

    assert "returns only the requested `target_view`" in api
    assert "returns only the view requested by `CacheSpec.target_view`" in cache
