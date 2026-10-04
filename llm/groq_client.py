"""
Groq LLM client — drop-in replacement for OpenAILLMClient using the
Groq cloud API (openai-compatible endpoint).

Default model: openai/gpt-oss-20b (as selected by the user).

Set your key:
  $env:GROQ_API_KEY="gsk_..."
Or store in api_key_env.env
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Optional


def _load_env_file_if_needed() -> None:
    env_paths = [
        Path("api_key_env.env"),
        Path(__file__).resolve().parents[1] / "api_key_env.env",
        Path(".env"),
    ]
    for p in env_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            os.environ[k] = v
            except Exception:
                pass


class GroqLLMClient:
    DEFAULT_MODEL = "openai/gpt-oss-20b"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: Optional[str] = None, api_key_env: str = "GROQ_API_KEY") -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK is required for Groq. Run:\n  pip install -U openai"
            ) from e

        _load_env_file_if_needed()

        raw_key = os.environ.get(api_key_env)
        if raw_key:
            raw_key = raw_key.strip().strip('"').strip("'").strip()
            raw_key = "".join(raw_key.split())
        api_key = raw_key or ""

        if not api_key:
            raise RuntimeError(
                f"No Groq API key found in environment variable {api_key_env}.\n"
                "Get a key at https://console.groq.com/keys, then set it:\n"
                f'  $env:{api_key_env}="gsk_..."'
            )

        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        self._model = model or os.environ.get("GROQ_MODEL") or self.DEFAULT_MODEL

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Calls Groq with system + user prompt constrained to JSON output.
        Raises ValueError if output is not valid JSON; RuntimeError on auth errors."""
        from openai import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError

        max_attempts = 4
        raw_text = ""
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
                    "Groq rejected the API key (401 Unauthorized).\n"
                    "Verify your key at https://console.groq.com/keys and re-set:\n"
                    '  $env:GROQ_API_KEY="gsk_..."'
                ) from e
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                if attempt == max_attempts:
                    raise
                # Check if Groq gave a specific retry duration in the error message
                delay = 2 ** (attempt - 1)
                err_str = str(e)
                import re as _re
                match = _re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    delay = min(15.0, float(match.group(1)) + 0.5)
                elif "TPD" in err_str or "tokens per day" in err_str:
                    print(f"[groq] Daily token quota exceeded: {err_str}")
                    raise
                print(f"[groq] {type(e).__name__} on attempt {attempt}/{max_attempts}; retrying in {delay:.1f}s...")
                time.sleep(delay)

        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Groq did not return valid JSON: {raw_text!r}") from e


if __name__ == "__main__":
    print("Testing Groq client...")
    try:
        client = GroqLLMClient()
        res = client.complete_json(
            'Respond ONLY with valid JSON: {"status": "ok", "model": "..."}',
            "Say hello and confirm JSON mode works."
        )
        print("Success! Response:", res)
    except Exception as e:
        print(f"Error: {e}")
