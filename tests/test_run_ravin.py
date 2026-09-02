from types import SimpleNamespace
from typing import cast
import scripts.run_ravin as run_ravin_module
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.service.bootstrap import (
    PolicyLoadProgress,
)

def test_cli_builds_service_through_shared_bootstrap(
    monkeypatch,
):
    service = cast(
        RavinAnswerService,
        object(),
    )

    runtime_config = SimpleNamespace(
        generation=SimpleNamespace(
            provider="test-generation-provider",
            model="test-generation-model",
        )
    )

    quality_config = SimpleNamespace(
        status="development-not-validated"
    )

    captured = {}

    monkeypatch.setattr(
        run_ravin_module,
        "load_runtime_provider_config",
        lambda: runtime_config,
    )

    monkeypatch.setattr(
        run_ravin_module,
        "load_answer_quality_config",
        lambda: quality_config,
    )

    def fake_create_service(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        callback = kwargs[
            "on_policy_loaded"
        ]

        callback(
            PolicyLoadProgress(
                policy_id="220",
                title=(
                    "Academic Progression "
                    "Review Policy"
                ),
                chunk_count=33,
            )
        )

        return service

    monkeypatch.setattr(
        run_ravin_module,
        "create_current_policy_ravin_service",
        fake_create_service,
    )

    result = (
        run_ravin_module.build_ravin_service()
    )

    assert result is service

    assert (
        captured["runtime_config"]
        is runtime_config
    )

    assert (
        captured[
            "answer_quality_config"
        ]
        is quality_config
    )

    assert callable(
        captured["on_policy_loaded"]
    )