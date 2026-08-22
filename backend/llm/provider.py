from typing import Protocol

class LanguageModelProvider(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...

def generate_text(
    provider: LanguageModelProvider,
    system_prompt: str,
    user_prompt: str,
) -> str:
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