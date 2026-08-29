"""
Define the framework-neutral language-model provider contract.

The protocol in this module gives grounded generation a minimal text
generation interface without coupling the generation layer to Ollama
or another model vendor.

Provider neutrality allows the concrete language-model implementation
to change while preserving the evidence-first RAVIN control pipeline.
"""

from typing import Protocol

class LanguageModelProvider(Protocol):
    """Define the framework-neutral text-generation provider contract.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate text from separate system and user prompts.
        """
        ...

def generate_text(
    provider: LanguageModelProvider,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Validate prompts, invoke the configured language model, and reject empty output.
    """
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()

    if not system_prompt:
        raise ValueError(
            "System prompt cannot be empty."
        )

    if not user_prompt:
        raise ValueError(
            "User prompt cannot be empty."
        )

    response = provider.generate(
        system_prompt,
        user_prompt,
    ).strip()

    if not response:
        raise ValueError(
            "Language model returned an "
            "empty response."
        )

    return response