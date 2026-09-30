"""
Shared Gemini client — used by the Controller's Stage 2 confirm (M1) and
the Decomposer (M2). Both are on or near the critical latency path, so this
is deliberately thin: one method, JSON in, JSON out.

Decision log note: "Gemini 1.5 Flash" (your original choice) has been
retired by Google. This defaults to "gemini-2.5-flash" instead. Verify the
current recommended Flash-tier model name yourself against
https://ai.google.dev/gemini-api/docs/models before you build further on
this — model availability changes faster than this file will be updated.

SDK note: this uses `google-genai` (the current, actively developed SDK —
`import google.generativeai as genai` is the OLD SDK and is not what this
code uses). Install with: pip install google-genai
"""

from __future__ import annotations

import json
import os
from typing import Optional


class GeminiLLMClient:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model: Optional[str] = None, api_key_env: str = "GEMINI_API_KEY") -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. Run:\n  pip install google-genai"
            ) from e

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in environment variable {api_key_env}.\n"
                "Get a free key at https://aistudio.google.com/apikey, then:\n"
                f"  export {api_key_env}=your_key_here      (macOS/Linux)\n"
                f"  $env:{api_key_env}=\"your_key_here\"      (Windows PowerShell)"
            )

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Calls Gemini with a system instruction + user prompt, constrained
        to return JSON, and returns the parsed dict. Raises ValueError if
        the model's output isn't valid JSON — callers should catch this and
        decide their own fallback (e.g. WAIT for the controller)."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Gemini did not return valid JSON: {response.text!r}") from e
