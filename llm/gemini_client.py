"""
Shared Gemini client — used by the Controller's Stage 2 confirm (M1) and
the Decomposer (M2). Both are on or near the critical latency path, so this
is deliberately thin: one method, JSON in, JSON out.

FIX LOG (2026-09-30):
  1. DEFAULT_MODEL "gemini-3.0-flash" was not a real model id -> 404 waiting
     behind the auth error. Now "gemini-3.8-flash" (current stable Flash).
  2. API key value is sanitized (stray quotes/whitespace/newlines stripped) —
     the classic Windows `set VAR="..."` / copy-paste failure mode.
  3. 401/403 responses now raise a short, actionable error instead of a
     five-screen traceback. Root cause of the 401 you hit: Google started
     rejecting old-format "Standard" API keys in Sept 2026; create a FRESH
     key at https://aistudio.google.com/apikey and run the current SDK:
     pip install -U google-genai
  4. Added a __main__ self-test: `python llm/gemini_client.py` checks just
     auth + JSON, without booting the whole pipeline.

SDK note: this uses `google-genai` (the current, actively developed SDK).
Install/upgrade with: pip install -U google-genai
"""

from __future__ import annotations

import json
import os
from typing import Optional


class GeminiLLMClient:
    # CHANGED — was: DEFAULT_MODEL = "gemini-3.0-flash"  (not a real model id)
    # Current stable Flash as of Sept 2026. "gemini-3.7-flash" also valid.
    DEFAULT_MODEL = "gemini-3.8-flash"

    def __init__(self, model: Optional[str] = None, api_key_env: str = "GEMINI_API_KEY") -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. Run:\n  pip install -U google-genai"
            ) from e

        # CHANGED — was: api_key = os.environ.get(api_key_env)
        # Sanitize: strip surrounding whitespace AND surrounding quotes and any
        # embedded whitespace/newlines. Windows CMD `set GEMINI_API_KEY="..."`
        # stores the quotes as part of the value -> Google returns 401.
        raw_key = os.environ.get(api_key_env)
        if raw_key:
            raw_key = raw_key.strip().strip('"').strip("'").strip()
            raw_key = "".join(raw_key.split())
        api_key = raw_key or ""

        if not api_key:
            raise RuntimeError(
                f"No API key found in environment variable {api_key_env}.\n"
                "IMPORTANT: since Sept 2026 Google rejects old-format 'Standard' keys —\n"
                "create a FRESH key at https://aistudio.google.com/apikey, then:\n"
                f'  $env:{api_key_env}="paste_new_key_here"   (Windows PowerShell, this window only)\n'
                f'  setx {api_key_env} "paste_new_key_here"    (persist for NEW terminals)\n'
                "Also make sure the SDK is current:\n"
                "  pip install -U google-genai"
            )

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        # Free tier allows ~5 requests/min PER MODEL. Stay under it proactively
        # (retries count too). Raise with $env:GEMINI_RPM="60" on a paid key.
        from collections import deque
        self._rpm = int(os.environ.get("GEMINI_RPM", "5"))
        self._call_times: "deque[float]" = deque()

    def _throttle(self) -> None:
        """Sliding-window limiter: block only when the last 60s already
        contain `_rpm` calls."""
        import time
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] >= 60.0:
            self._call_times.popleft()
        if len(self._call_times) >= self._rpm:
            wait = 60.0 - (now - self._call_times[0]) + 0.25
            print(f"[gemini] rate limit window full ({self._rpm}/min); waiting {wait:.1f}s...")
            time.sleep(max(wait, 0.0))
        self._call_times.append(time.monotonic())

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Calls Gemini with a system instruction + user prompt, constrained
        to return JSON, and returns the parsed dict. Raises ValueError if
        the model's output isn't valid JSON — callers should catch this and
        decide their own fallback (e.g. WAIT for the controller)."""
        # CHANGED — added this import (was: only `from google.genai import types`)
        from google.genai import errors as genai_errors
        from google.genai import types

        # Transient failures (503 overloaded, 500/504, 429 rate limit) are
        # retried with exponential backoff before giving up.
        import time
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                self._throttle()
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                    ),
                )
                break
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                code = getattr(e, "code", None)
                if code in (401, 403):
                    raise RuntimeError(
                        f"Gemini rejected the API key (HTTP {e.code}). Fix:\n"
                        "  1. Create a NEW key at https://aistudio.google.com/apikey\n"
                        "  2. python -m pip install -U google-genai\n"
                        "  3. Re-set it in THIS PowerShell window:\n"
                        '       $env:GEMINI_API_KEY="paste_new_key_here"'
                    ) from e
                if code not in (429, 500, 503, 504) or attempt == max_attempts:
                    raise
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                if code == 429:
                    # Google states the wait: "Please retry in 9.99s"
                    import re
                    m = re.search(r"retry in ([\d.]+)s", str(e))
                    if m:
                        delay = float(m.group(1)) + 1.0
                    else:
                        delay = max(delay, 10)
                print(f"[gemini] {code} on attempt {attempt}/{max_attempts}; retrying in {delay:.0f}s...")
                time.sleep(delay)

        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Gemini did not return valid JSON: {response.text!r}") from e


# CHANGED — added this whole block (file previously ended at the except above).
# Self-test:  python llm/gemini_client.py
if __name__ == "__main__":
    client = GeminiLLMClient()
    print(f"Model in use: {client._model}")
    print("Key is accepted. Models this key can see:")
    for m in client._client.models.list():
        name = getattr(m, "name", "")
        if "flash" in name.lower():
            print("  -", name)
    result = client.complete_json(
        "You answer with strict JSON only.",
        'Reply with exactly this JSON and nothing else: {"ok": true}',
    )
    print("Auth + JSON round-trip OK:", result)
