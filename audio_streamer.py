"""
Microphone Audio Streamer for Streaming Live RAG.

Captures live microphone audio via sounddevice, performs local streaming
transcription using faster-whisper (base.en), and yields real-time
TranscriptChunk objects directly into the M1 Retrieval Controller.

Features:
  - Automatic Gain Control (AGC) & Volume Normalization to handle quiet/laptop mics
  - Live ASCII VU meter to give immediate visual feedback on microphone levels
  - Silero VAD filtering to completely prevent silence hallucinations ("Got it", "Thank you")
  - Push-to-Talk / Press-Enter-to-Finish support via non-blocking keyboard input
  - Device enumeration and selection (laptop mic array vs Bluetooth headset/buds)
"""

from __future__ import annotations

import os
import queue
import re
import sys
import time
from typing import Generator, List, Optional

import numpy as np

from contracts import TranscriptChunk

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import msvcrt
    WINDOWS_KEYBOARD = True
except ImportError:
    WINDOWS_KEYBOARD = False


DOMAIN_PROMPT = (
    "The user is asking questions about venue booking, seated capacity, "
    "cancellation policies, catering options, workshops in Pune, and travel reimbursement."
)


def list_microphones() -> List[dict]:
    """Returns a list of all available audio input devices."""
    if not AUDIO_AVAILABLE:
        return []
    devices = sd.query_devices()
    inputs = []
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else 0
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append({
                "index": idx,
                "name": d.get("name", "Unknown"),
                "channels": d.get("max_input_channels", 1),
                "is_default": (idx == default_in),
            })
    return inputs


def clean_transcript(text: str) -> str:
    """Removes common Whisper silence hallucinations and repetitive conversational fillers."""
    t = text.strip()
    hallucination_patterns = [
        r"^(got it\.?\s*)+",
        r"^(thank you\.?\s*)+",
        r"^(thanks for watching\.?\s*)+",
        r"^(you\.?\s*)+",
        r"^(bye\.?\s*)+",
        r"^(so\.?\s*)+",
    ]
    for p in hallucination_patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE).strip()
    return t


class MicrophoneStreamer:
    """
    Listens to live microphone audio and emits TranscriptChunk objects
    incrementally into the Streaming Live RAG pipeline.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        sample_rate: int = 16000,
        chunk_duration_sec: float = 0.25,
        device_index: Optional[int] = None,
    ) -> None:
        if not AUDIO_AVAILABLE:
            raise RuntimeError(
                "Microphone streaming requires 'sounddevice' and 'faster-whisper'.\n"
                "Install them with: pip install sounddevice faster-whisper"
            )
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_sec)
        self.chunk_duration_sec = chunk_duration_sec
        self.device_index = device_index
        self.last_full_transcript: str = ""

        # Print active input device
        try:
            active_info = sd.query_devices(self.device_index, "input") if self.device_index is not None else sd.query_devices(kind="input")
            dev_name = active_info.get("name", "Default")
            print(f"      Audio Input Device: {dev_name} (Index: {self.device_index if self.device_index is not None else 'Default'})", flush=True)
        except Exception:
            pass

        print(f"      Loading local Whisper model ({model_size})...", flush=True)
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("      Microphone and speech engine ready!", flush=True)

    def listen_and_stream(
        self,
        session_id: str,
        silence_timeout_sec: float = 2.0,
        max_duration_sec: float = 30.0,
        initial_silence_timeout_sec: float = 12.0,
    ) -> Generator[TranscriptChunk, None, None]:
        """
        Streams live speech from the microphone, yielding TranscriptChunk
        objects incrementally until the speaker finishes or presses Enter.
        """
        self.last_full_transcript = ""
        audio_q: queue.Queue[np.ndarray] = queue.Queue()

        def _audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_q.put(indata.copy())

        print("\n" + "=" * 68)
        print("  🎙️  MICROPHONE ACTIVE — Speak your question clearly into your mic!")
        print("  • Press [Enter] on your keyboard when finished (or pause for 2s)")
        print("  • Watch the Level meter below to verify your voice is detected")
        print("=" * 68 + "\n")

        audio_buffer: List[np.ndarray] = []
        seq = 0
        t_ms = 0.0
        start_time = time.time()
        last_speech_time: Optional[float] = None
        last_transcribe_time: float = 0.0
        has_spoken = False
        full_transcript = ""

        # Flush any stray keypresses in the stdin buffer
        if WINDOWS_KEYBOARD:
            while msvcrt.kbhit():
                msvcrt.getch()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=_audio_callback,
        ):
            while True:
                now = time.time()
                elapsed = now - start_time

                # Check for keyboard [Enter] or [Space] press to immediately finish
                user_pressed_done = False
                if WINDOWS_KEYBOARD and msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"\r", b"\n", b" "):
                        user_pressed_done = True

                if user_pressed_done or elapsed > max_duration_sec:
                    break

                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                audio_buffer.append(chunk)
                t_ms += round(self.chunk_duration_sec * 1000.0, 1)

                # Compute RMS energy for live meter
                rms = float(np.sqrt(np.mean(chunk**2)))
                # Log-scaled meter: 1e-5 (silence) to 0.02 (loud speech)
                meter_val = (np.log10(max(rms, 1e-5)) + 4.5) / 3.2
                bars = int(np.clip(meter_val * 10, 0, 10))
                meter_str = "=" * bars + "-" * (10 - bars)

                # Visual feedback line
                status_label = "Speaking" if rms > 0.0003 else "Listening"
                display_heard = f' | Heard: "{full_transcript}"' if full_transcript else ""
                sys.stdout.write(f"\r  🎙️ [{status_label}] Level: [{meter_str}]{display_heard}   ")
                sys.stdout.flush()

                # Speech threshold (sensitive enough for quiet laptop microphones)
                if rms > 0.0003:
                    has_spoken = True
                    last_speech_time = now

                # Streaming intermediate transcription:
                # Transcribe every ~0.8s once at least 1.0s of speech has accumulated
                total_duration = len(audio_buffer) * self.chunk_duration_sec
                if has_spoken and total_duration >= 1.0 and (now - last_transcribe_time > 0.8):
                    last_transcribe_time = now

                    # Boost faint audio to studio level (AGC)
                    flat_audio = np.concatenate(audio_buffer).flatten()
                    peak = float(np.max(np.abs(flat_audio)))
                    if peak > 0.0001:
                        normalized_audio = flat_audio / peak * 0.92
                    else:
                        normalized_audio = flat_audio

                    try:
                        segments, _ = self.model.transcribe(
                            normalized_audio,
                            beam_size=1,
                            language="en",
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                            condition_on_previous_text=False,
                            no_speech_threshold=0.5,
                            initial_prompt=DOMAIN_PROMPT,
                        )
                        raw_text = " ".join(s.text.strip() for s in segments).strip()
                        current_text = clean_transcript(raw_text)

                        # Emit new delta if text has grown
                        if current_text and len(current_text) > len(full_transcript):
                            delta = current_text[len(full_transcript):].strip()
                            full_transcript = current_text
                            seq += 1

                            yield TranscriptChunk(
                                session_id=session_id,
                                seq=seq,
                                t_ms=t_ms,
                                text_delta=delta,
                                is_final=False,
                            )
                    except Exception:
                        pass

                # Check for silence timeout after speech began
                if has_spoken and last_speech_time and (now - last_speech_time > silence_timeout_sec):
                    break

                # Initial timeout if user doesn't say anything
                if not has_spoken and elapsed > initial_silence_timeout_sec:
                    sys.stdout.write("\n\n  [Audio] No speech detected within 12 seconds.\n")
                    sys.stdout.flush()
                    return

        # Final high-precision transcription pass over the complete recording
        if audio_buffer:
            sys.stdout.write("\n  [ASR Processing final audio...]\r")
            sys.stdout.flush()

            flat_audio = np.concatenate(audio_buffer).flatten()
            peak = float(np.max(np.abs(flat_audio)))
            if peak > 0.0001:
                normalized_audio = flat_audio / peak * 0.92
            else:
                normalized_audio = flat_audio

            try:
                segments, _ = self.model.transcribe(
                    normalized_audio,
                    beam_size=2,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                    condition_on_previous_text=False,
                    no_speech_threshold=0.4,
                    initial_prompt=DOMAIN_PROMPT,
                )
                raw_final = " ".join(s.text.strip() for s in segments).strip()
                final_text = clean_transcript(raw_final)
            except Exception as e:
                final_text = full_transcript

            if final_text:
                full_transcript = final_text

        self.last_full_transcript = full_transcript

        # Emit the final end-of-speech chunk
        if full_transcript:
            seq += 1
            yield TranscriptChunk(
                session_id=session_id,
                seq=seq,
                t_ms=t_ms,
                text_delta="",
                is_final=True,
            )
            print(f"\n  ✅ [ASR Completed]: \"{full_transcript}\"\n")
        else:
            print("\n  ⚠️  [ASR]: No clear speech recognized. Please try speaking closer to the microphone.\n")


if __name__ == "__main__":
    if not AUDIO_AVAILABLE:
        print("Please install: pip install sounddevice faster-whisper")
        sys.exit(1)

    print("--- Available Input Devices ---")
    for d in list_microphones():
        def_tag = " (DEFAULT)" if d["is_default"] else ""
        print(f"  [{d['index']}] {d['name']}{def_tag}")
    print("-------------------------------\n")

    streamer = MicrophoneStreamer()
    for c in streamer.listen_and_stream(session_id="test-session"):
        print(f"CHUNK -> seq={c.seq}, is_final={c.is_final}, delta={c.text_delta!r}")