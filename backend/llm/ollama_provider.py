import json
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.request import (
    Request,
    urlopen,
)

_DEFAULT_BASE_URL = (
    "http://localhost:11434"
)
_DEFAULT_TIMEOUT_SECONDS = 300.0

class OllamaLanguageModelProvider:
    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = (
            _DEFAULT_TIMEOUT_SECONDS
        ),
    ) -> None:
        model_name = model_name.strip()
        base_url = base_url.strip().rstrip("/")

        if not model_name:
            raise ValueError(
                "Ollama model name cannot "
                "be empty."
            )

        if not base_url:
            raise ValueError(
                "Ollama base URL cannot "
                "be empty."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "Ollama timeout must be "
                "greater than zero."
            )

        self._model_name = model_name
        self._base_url = base_url
        self._timeout_seconds = (
            timeout_seconds
        )

    @property
    def model_name(
        self,
    ) -> str:
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        payload = {
            "model": self._model_name,
            "messages": (
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ),
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        request = Request(
            url=(
                f"{self._base_url}/api/chat"
            ),
            data=json.dumps(
                payload
            ).encode(
                "utf-8"
            ),
            headers={
                "Content-Type": (
                    "application/json"
                ),
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                response_body = (
                    response.read()
                )

        except HTTPError as error:
            raise RuntimeError(
                "Ollama returned an HTTP error."
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Ollama could not be reached."
            ) from error

        except TimeoutError as error:
            raise RuntimeError(
                "Ollama request timed out."
            ) from error

        try:
            response_data = json.loads(
                response_body.decode(
                    "utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(
                "Ollama returned an invalid "
                "response."
            ) from error

        message = response_data.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "Ollama response does not "
                "contain a message."
            )

        content = message.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):
            raise RuntimeError(
                "Ollama response does not "
                "contain generated text."
            )

        return content