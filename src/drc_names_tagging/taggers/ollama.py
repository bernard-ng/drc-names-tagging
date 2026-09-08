from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ollama import Client, ResponseError

from drc_names_tagging.models import Name, Tag, Token, tokens

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                    "tag": {"type": "string", "enum": ["native", "foreign"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["token", "tag", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["components"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Ollama:
    """Structured-output tagger backed by a local Ollama model."""

    model: str = "mistral:7b"
    url: str = "http://localhost:11434"
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return "ollama"

    def tag(self, name: str) -> Name:
        source = tokens(name)
        if not source:
            return Name(name, ())

        response = self._chat(name, source)
        values = response.get("components")
        if not isinstance(values, list):
            raise TypeError("Ollama response must contain a components array.")
        if len(values) != len(source):
            raise ValueError(
                f"Ollama returned {len(values)} components for {len(source)} tokens."
            )

        annotations: list[Token] = []
        for index, (token, value) in enumerate(zip(source, values)):
            if not isinstance(value, dict):
                raise TypeError("Each Ollama component must be an object.")
            returned = str(value.get("token", ""))
            if returned.strip().lower() != token.strip().lower():
                raise ValueError(
                    f"Ollama changed token {index}: expected '{token}', received '{returned}'."
                )
            confidence = float(value.get("confidence", 0.0))
            if not 0 <= confidence <= 1:
                raise ValueError("Ollama confidence must be between 0 and 1.")
            annotations.append(Token(index, token, Tag(str(value.get("tag", ""))), confidence))
        return Name(name, tuple(annotations))

    def _chat(self, name: str, source: list[str]) -> dict[str, Any]:
        prompt = (
            "Annotate the tokens in this personal name by position. "
            "Assign exactly one tag, native or foreign, to every token occurrence. "
            "Foreign means a token that is not native to the DRC/African naming context; "
            "it is not a surname label.\n"
            "Never deduplicate, merge, omit, or reorder tokens. Repeated spellings are "
            "separate occurrences and must each receive their own component and tag. "
            "All tokens may legitimately receive the native tag, and all tokens may "
            "legitimately receive the foreign tag. Echo each token exactly and return "
            "the same number of components as input tokens. Return a confidence from 0 "
            "to 1 for every tag.\n"
            "For example, for bope ndjondo ndjondo, return three separate components "
            "in that order and tag all three as native; the two ndjondo components "
            "must not be collapsed into one.\n\n"
            f"Name: {name}\nTokens: {json.dumps(source, ensure_ascii=False)}"
        )
        try:
            response = Client(host=self.url, timeout=self.timeout).chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful linguistic annotation assistant. "
                            "Produce one position-preserving annotation for every input "
                            "token, including repeated tokens."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                format=SCHEMA,
                options={"temperature": 0},
            )
        except ResponseError as error:
            raise RuntimeError(
                f"Ollama request failed for {self.model} at {self.url}: {error}"
            ) from error
        except OSError as error:
            raise RuntimeError(f"Could not reach Ollama at {self.url}: {error}") from error

        try:
            content = response.message.content
            if not isinstance(content, str):
                raise TypeError("Ollama response content must be text.")
            parsed = json.loads(content)
        except (TypeError, ValueError) as error:
            raise ValueError("Ollama returned invalid structured tagging output.") from error
        if not isinstance(parsed, dict):
            raise TypeError("Ollama structured output must be a JSON object.")
        return parsed
