from types import SimpleNamespace
from typing import cast
import pytest
import backend.service.bootstrap as bootstrap_module

from backend.ingestion.models import (
    PolicyChunk,
)
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.service.bootstrap import (
    CURRENT_POLICY_LINKS,
    PolicyLoadProgress,
    acquire_current_policy_chunks,
    create_current_policy_ravin_service,
)

def test_current_policy_corpus_contains_expected_policies():
    assert tuple(
        link.policy_id
        for link in CURRENT_POLICY_LINKS
    ) == (
        "208",
        "220",
        "76",
        "420",
        "169",
        "340",
    )

def test_acquisition_aggregates_chunks_and_reports_progress(
    monkeypatch,
):
    chunk = cast(
        PolicyChunk,
        object(),
    )

    acquired_links = []
    progress: list[
        PolicyLoadProgress
    ] = []

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

        return SimpleNamespace(
            policy_id=link.policy_id,
            title=link.title,
        )

    def fake_process_policy(
        raw_policy,
    ):
        return SimpleNamespace(
            chunks=(
                chunk,
            ),
            error=None,
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

    assert len(chunks) == len(
        CURRENT_POLICY_LINKS
    )

    assert all(
        item is chunk
        for item in chunks
    )

    assert tuple(
        link
        for link, _ in acquired_links
    ) == CURRENT_POLICY_LINKS

    assert all(
        timeout == 20.0
        for _, timeout in acquired_links
    )

    assert tuple(
        item.policy_id
        for item in progress
    ) == tuple(
        link.policy_id
        for link in CURRENT_POLICY_LINKS
    )

    assert all(
        item.chunk_count == 1
        for item in progress
    )

def test_acquisition_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match="timeout must be greater than zero",
    ):
        acquire_current_policy_chunks(
            timeout_seconds=0,
        )

def test_failed_policy_ingestion_stops_bootstrap(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_module,
        "acquire_policy",
        lambda link, timeout_seconds: (
            SimpleNamespace()
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
        match="Policy ingestion failed for 208",
    ):
        acquire_current_policy_chunks()

def test_service_bootstrap_delegates_to_service_composition(
    monkeypatch,
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
        is None
    )

    assert (
        captured["kwargs"][
            "answer_quality_config"
        ]
        is None
    )