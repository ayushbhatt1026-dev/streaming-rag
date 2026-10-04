"""
Shared OpenAI client — used by the Controller's Stage 2 confirm (M1),
the Decomposer (M2), and later the Synthesizer (M5).

Designed for low latency and strict JSON outputs using gpt-4o-mini.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


class OpenAILLMClient:
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY") -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is not installed. Run:\n  pip install -U openai"
            ) from e

        raw_key = os.environ.get(api_key_env)
        if raw_key:
            raw_key = raw_key.strip().strip('"').strip("'").strip()
            raw_key = "".join(raw_key.split())
        api_key = raw_key or ""

        if not api_key:
            raise RuntimeError(
                f"No API key found in environment variable {api_key_env}.\n"
                "Please set it in your PowerShell window:\n"
                f'  $env:{api_key_env}="sk-your-openai-key-here"\n'
                f'Or persist it with: setx {api_key_env} "sk-your-openai-key-here"'
            )

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Calls OpenAI with system prompt + user prompt, constrained to return JSON,
        and returns the parsed dict. Raises ValueError if output isn't valid JSON."""
        from openai import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_text = response.choices[0].message.content or ""
                break
            except AuthenticationError as e:
                raise RuntimeError(
                    f"OpenAI rejected the API key (401 Unauthorized).\n"
                    "Please verify your key at https://platform.openai.com/api-keys and re-set:\n"
                    '  $env:OPENAI_API_KEY="sk-..."'
                ) from e
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                if attempt == max_attempts:
                    raise
                delay = 2 ** (attempt - 1)
                print(f"[openai] {type(e).__name__} on attempt {attempt}/{max_attempts}; retrying in {delay}s...")
                time.sleep(delay)

        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"OpenAI did not return valid JSON: {raw_text!r}") from e


if __name__ == "__main__":
    print("Testing OpenAI client...")
    try:
        client = OpenAILLMClient()
        res = client.complete_json(
            "Respond ONLY with valid JSON: {\"status\": \"ok\", \"model\": \"...\"}",
            "Say hello and confirm JSON mode works."
        )
        print("Success! Response from OpenAI:", res)
    except Exception as e:
        print(f"Error testing OpenAI client: {e}")
