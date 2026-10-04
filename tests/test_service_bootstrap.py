from types import SimpleNamespace
from typing import cast
import pytest
import backend.service.bootstrap as bootstrap_module
from backend.ingestion.acquisition import (
    PolicyAccessRestrictedError,
    PolicyLink,
)
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.service.bootstrap import (
    PolicyLoadProgress,
    acquire_current_policy_chunks,
    create_current_policy_ravin_service,
)
from backend.retrieval.cache import (
    RetrievalCacheInvalidError,
    RetrievalCacheStaleError,
)
from pathlib import Path

def _link(
    policy_id: str,
    title: str,
) -> PolicyLink:
    return PolicyLink(
        policy_id=policy_id,
        title=title,
        url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
    )

def test_stale_cache_requires_explicit_refresh(
    monkeypatch,
    tmp_path: Path,
):
    cache_path = (
        tmp_path
        / "cache.json.gz"
    )

    cache_path.write_text(
        "stale",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        bootstrap_module,
        "load_retrieval_cache",
        lambda *args, **kwargs: (
            (_ for _ in ())
            .throw(
                RetrievalCacheStaleError(
                    "Retrieval cache is stale."
                )
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="RefreshPolicyCache",
    ):
        create_current_policy_ravin_service(
            cache_path=cache_path,
        )

def test_acquisition_discovers_policies_and_reports_progress(
    monkeypatch,
):
    chunk = cast(
        PolicyChunk,
        object(),
    )

    links = (
        _link(
            "208",
            "Academic Dress Policy",
        ),
        _link(
            "999",
            "Old Example Policy",
        ),
        _link(
            "268",
            "Investment Policy",
        ),
    )

    browse_calls = []
    acquired_links = []
    progress: list[
        PolicyLoadProgress
    ] = []

    def fake_fetch_html(
        url,
        timeout_seconds,
    ):
        browse_calls.append(
            (
                url,
                timeout_seconds,
            )
        )

        return "<html>Browse</html>"

    def fake_acquire_policy(
        link,
        timeout_seconds,
    ):
        acquired_links.append(
            (
                link,
                timeout_seconds,
            )
        )

        if link.policy_id == "268":
            raise PolicyAccessRestrictedError(
                "Policy source requires "
                "authenticated access."
            )

        return SimpleNamespace(
            policy_id=link.policy_id,
            title=link.title,
        )

    def fake_process_policy(
        raw_policy,
    ):
        if raw_policy.policy_id == "999":
            return SimpleNamespace(
                chunks=(),
                error=(
                    "Policy is not current."
                ),
            )

        return SimpleNamespace(
            chunks=(
                chunk,
            ),
            error=None,
        )

    monkeypatch.setattr(
        bootstrap_module,
        "fetch_html",
        fake_fetch_html,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_policy_links",
        lambda html: links,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_policy",
        fake_acquire_policy,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "process_policy",
        fake_process_policy,
    )

    chunks = acquire_current_policy_chunks(
        timeout_seconds=20.0,
        on_policy_loaded=progress.append,
    )

    assert chunks == (
        chunk,
    )

    assert browse_calls == [
        (
            bootstrap_module.BROWSE_URL,
            20.0,
        ),
    ]

    assert tuple(
        link.policy_id
        for link, _ in acquired_links
    ) == (
        "208",
        "999",
        "268",
    )

    assert all(
        timeout == 20.0
        for _, timeout in acquired_links
    )

    assert tuple(
        (
            item.policy_id,
            item.status,
            item.chunk_count,
        )
        for item in progress
    ) == (
        (
            "208",
            "current",
            1,
        ),
        (
            "999",
            "not_current",
            0,
        ),
        (
            "268",
            "restricted",
            0,
        ),
    )

def test_acquisition_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match=(
            "timeout must be greater than zero"
        ),
    ):
        acquire_current_policy_chunks(
            timeout_seconds=0,
        )

def test_empty_discovery_stops_bootstrap(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_module,
        "fetch_html",
        lambda url, timeout_seconds: (
            "<html></html>"
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_policy_links",
        lambda html: (),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "No policy links were discovered"
        ),
    ):
        acquire_current_policy_chunks()

def test_failed_policy_ingestion_stops_bootstrap(
    monkeypatch,
):
    links = (
        _link(
            "208",
            "Academic Dress Policy",
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "fetch_html",
        lambda url, timeout_seconds: (
            "<html>Browse</html>"
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_policy_links",
        lambda html: links,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_policy",
        lambda link, timeout_seconds: (
            SimpleNamespace(
                policy_id=link.policy_id,
                title=link.title,
            )
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "process_policy",
        lambda raw_policy: (
            SimpleNamespace(
                chunks=(),
                error="not accepted",
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Policy ingestion failed for 208"
        ),
    ):
        acquire_current_policy_chunks()

def test_unexpected_restricted_policy_stops_bootstrap(
    monkeypatch,
):
    chunk = cast(
        PolicyChunk,
        object(),
    )

    links = (
        _link(
            "208",
            "Academic Dress Policy",
        ),
        _link(
            "999",
            "Unexpected Restricted Policy",
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "fetch_html",
        lambda url, timeout_seconds: (
            "<html>Browse</html>"
        ),
    )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_policy_links",
        lambda html: links,
    )

    def fake_acquire_policy(
        link,
        timeout_seconds,
    ):
        if link.policy_id == "999":
            raise PolicyAccessRestrictedError(
                "Policy source requires "
                "authenticated access."
            )

        return SimpleNamespace(
            policy_id=link.policy_id,
            title=link.title,
        )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_policy",
        fake_acquire_policy,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "process_policy",
        lambda raw_policy: (
            SimpleNamespace(
                chunks=(
                    chunk,
                ),
                error=None,
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Unexpected restricted policies: 999"
        ),
    ):
        acquire_current_policy_chunks()

def test_service_bootstrap_delegates_to_service_composition(
    monkeypatch,
    tmp_path: Path,
):
    chunks = (
        cast(
            PolicyChunk,
            object(),
        ),
    )

    service = cast(
        RavinAnswerService,
        object(),
    )

    captured = {}

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_current_policy_chunks",
        lambda **kwargs: chunks,
    )

    def fake_create_service(
        supplied_chunks,
        **kwargs,
    ):
        captured["chunks"] = (
            supplied_chunks
        )

        captured["kwargs"] = kwargs

        return service

    monkeypatch.setattr(
        bootstrap_module,
        "create_ravin_answer_service",
        fake_create_service,
    )

    result = (
        create_current_policy_ravin_service(
            timeout_seconds=25.0,
            cache_path=(
                tmp_path
                / "cache.json.gz"
            ),
        )
    )

    assert result is service

    assert (
        captured["chunks"]
        is chunks
    )

    assert (
        captured["kwargs"][
            "runtime_config"
        ]
        is not None
    )

    assert (
        captured["kwargs"][
            "answer_quality_config"
        ]
        is None
    )

def test_service_bootstrap_uses_compatible_cache(
    monkeypatch,
    tmp_path: Path,
):
    chunk = cast(
        PolicyChunk,
        object(),
    )

    indexed_chunk = SimpleNamespace(
        chunk=chunk,
    )

    cached_corpus = SimpleNamespace(
        metadata=SimpleNamespace(
            policy_count=1,
            chunk_count=1,
            created_at_utc=(
                "2026-09-26T00:00:00+00:00"
            ),
            restricted_policy_ids=(
                "268",
            ),
        ),
        indexed_chunks=(
            indexed_chunk,
        ),
    )

    cache_path = (
        tmp_path
        / "cache.json.gz"
    )

    cache_path.write_text(
        "placeholder",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        bootstrap_module,
        "load_retrieval_cache",
        lambda *args, **kwargs: (
            cached_corpus
        ),
    )

    def fail_if_acquired(
        **kwargs,
    ):
        raise AssertionError(
            "Live policy acquisition "
            "should not run on a cache hit."
        )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_current_policy_chunks",
        fail_if_acquired,
    )

    service = cast(
        RavinAnswerService,
        object(),
    )

    captured = {}

    def fake_create_service(
        supplied_chunks,
        **kwargs,
    ):
        captured["chunks"] = (
            supplied_chunks
        )

        captured["kwargs"] = kwargs

        return service

    monkeypatch.setattr(
        bootstrap_module,
        "create_ravin_answer_service",
        fake_create_service,
    )

    result = (
        create_current_policy_ravin_service(
            cache_path=cache_path,
        )
    )

    assert result is service

    assert captured["chunks"] == (
        chunk,
    )

    assert (
        captured["kwargs"][
            "indexed_chunks"
        ]
        is cached_corpus.indexed_chunks
    )


def test_service_bootstrap_refresh_bypasses_cache_and_saves_index(
    monkeypatch,
    tmp_path: Path,
):
    chunk = cast(
        PolicyChunk,
        object(),
    )

    indexed_chunks = (
        SimpleNamespace(
            chunk=chunk,
        ),
    )

    cache_path = (
        tmp_path
        / "cache.json.gz"
    )

    cache_path.write_text(
        "old cache",
        encoding="utf-8",
    )

    def fail_if_cache_loaded(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Existing cache should not be "
            "loaded during forced refresh."
        )

    monkeypatch.setattr(
        bootstrap_module,
        "load_retrieval_cache",
        fail_if_cache_loaded,
    )

    def fake_acquire(
        *,
        timeout_seconds,
        on_policy_loaded,
    ):
        assert (
            on_policy_loaded
            is not None
        )

        on_policy_loaded(
            PolicyLoadProgress(
                policy_id="208",
                title=(
                    "Academic Dress Policy"
                ),
                chunk_count=1,
                status="current",
            )
        )

        on_policy_loaded(
            PolicyLoadProgress(
                policy_id="268",
                title=(
                    "Investment Policy"
                ),
                chunk_count=0,
                status="restricted",
            )
        )

        return (
            chunk,
        )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_current_policy_chunks",
        fake_acquire,
    )

    saved = {}

    def fake_save_cache(
        path,
        supplied_indexed_chunks,
        **kwargs,
    ):
        saved["path"] = path
        saved["indexed_chunks"] = (
            supplied_indexed_chunks
        )
        saved["kwargs"] = kwargs

        return SimpleNamespace(
            policy_count=1,
            chunk_count=1,
            restricted_policy_ids=(
                "268",
            ),
        )

    monkeypatch.setattr(
        bootstrap_module,
        "save_retrieval_cache",
        fake_save_cache,
    )

    service = cast(
        RavinAnswerService,
        object(),
    )

    def fake_create_service(
        supplied_chunks,
        **kwargs,
    ):
        assert supplied_chunks == (
            chunk,
        )

        callback = kwargs[
            "on_index_ready"
        ]

        callback(
            indexed_chunks
        )

        return service

    monkeypatch.setattr(
        bootstrap_module,
        "create_ravin_answer_service",
        fake_create_service,
    )

    result = (
        create_current_policy_ravin_service(
            cache_path=cache_path,
            refresh_cache=True,
        )
    )

    assert result is service

    assert (
        saved["path"]
        == cache_path
    )

    assert (
        saved["indexed_chunks"]
        == indexed_chunks
    )

    assert (
        saved["kwargs"][
            "restricted_policy_ids"
        ]
        == ("268",)
    )


def test_invalid_cache_does_not_silently_trigger_live_rebuild(
    monkeypatch,
    tmp_path: Path,
):
    cache_path = (
        tmp_path
        / "cache.json.gz"
    )

    cache_path.write_text(
        "invalid",
        encoding="utf-8",
    )

    def fail_cache_load(
        *args,
        **kwargs,
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache checksum "
            "validation failed."
        )

    monkeypatch.setattr(
        bootstrap_module,
        "load_retrieval_cache",
        fail_cache_load,
    )

    def fail_if_acquired(
        **kwargs,
    ):
        raise AssertionError(
            "Invalid cache must not silently "
            "trigger a live rebuild."
        )

    monkeypatch.setattr(
        bootstrap_module,
        "acquire_current_policy_chunks",
        fail_if_acquired,
    )

    with pytest.raises(
        RetrievalCacheInvalidError,
        match="checksum",
    ):
        create_current_policy_ravin_service(
            cache_path=cache_path,
        )