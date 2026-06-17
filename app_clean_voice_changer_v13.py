import os
import random
import wave
import time
import queue
import tempfile
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

VOICE_LAB_URL = "https://elevenlabs.io/app/voice-lab"
LINK_GROUPS = {
    "Voice clone & conversion": [
        ("ElevenLabs Voice Lab", "https://elevenlabs.io/app/voice-lab"),
        ("Cartesia Voices", "https://play.cartesia.ai/voices"),
        ("Resemble AI Voices", "https://app.resemble.ai/hub/voices"),
    ],
    "Speaker identification": [
        ("TiTANet Speaker Verification", "https://huggingface.co/spaces/nithinraok/titanet-speaker-verification"),
        ("Phonexia Speaker Identification", "https://zwlb2f42bkx9.demo.cloud.phonexia.com/app/speaker-identification"),
    ],
    "Deepfake detection": [
        ("Phonexia Deepfake Detection", "https://zwlb2f42bkx9.demo.cloud.phonexia.com/app/deepfake-detection"),
        ("Resemble Detect", "https://app.resemble.ai/hub/detect"),
    ],
    "Watermarking": [
        ("Resemble Watermarking", "https://app.resemble.ai/hub/watermarking"),
    ],
}

MARKET_APP_ROWS = [
    {
        "name": "Voicemod Mobile",
        "platform": "iOS / Android",
        "intro": "Popular consumer soundboard / voice-effects ecosystem. Useful as a mobile reference for fun voice effects and soundboard-style UX.",
        "links": [
            ("App Store", "https://apps.apple.com/us/app/voicemod/id6473788687"),
            ("Google Play", "https://play.google.com/store/apps/details?id=net.voicemod.soundboard"),
        ],
        "image": "https://images.openai.com/static-rsc-4/qC2d1oPSJmJF6kI8ThhM3CCRXmm6s3lPOGJIvAG3qDEqLo5fbtTtNGrrTDRoOlrj_icHRSm9bbBEzsq_-n1XwLMH0ZUsWECQhhcZwpu7wjB6sylPKBcQZgkim9zxtz2z3D5vvTP70AOYSbo_fe5TB6eJfGekELMFliGNSpLx_-M?purpose=inline",
    },
    {
        "name": "Voice changer with effects",
        "platform": "Android",
        "intro": "Classic record-and-apply-effects app with simple sharing workflow and many non-AI voice effects.",
        "links": [
            ("Google Play", "https://play.google.com/store/apps/details?id=com.baviux.voicechanger"),
        ],
        "image": "https://tse4.mm.bing.net/th/id/OIP.sdzgfF-jeICiXiqSkeTlcgHaMW?r=0&pid=Api",
    },
    {
        "name": "MagicCall",
        "platform": "iOS / Android",
        "intro": "Real-time call voice changer with voice styles and background sounds. Useful as a live-call consumer comparison.",
        "links": [
            ("App Store", "https://apps.apple.com/us/app/magiccall-voice-changer-app/id1324524338"),
            ("Google Play", "https://play.google.com/store/apps/details?id=com.bng.magiccall"),
        ],
        "image": "https://images.openai.com/static-rsc-4/GqfJhl6Li3l4sOQ2nX2PrZTaECNBnt6lKr9lNgtB7ODtseFkj5PMVn0q_0G3RR85H2tsdm1Z--PQncJPrTIaGmnBTKOnaTU1hi2p9xf2GRbPKukBzc-8-v-37SVX4rwtYyccrdjMPytPHkd0Ey7LfGJQnaW7Ww6rteAFSAQB_wQ?purpose=inline",
    },
]

COMPARISON_AUDIO_FILES = {
    "eleven_tts": "Divij_2min_ElevenLabs_InstantClone_TTS.mp3",
    "eleven_sts": "Divij_2min_ElevenLabs_InstantClone_STS.mp3",
    "cartesia_tts": "Divij_10sec_Cartesia_InstantClone_TTS.wav",
    "cartesia_sts": "Divij_10sec_Cartesia_InstantClone_STS.mp3",
    "resemble_tts": "Divij_2min_Resemble_Clone_TTS.wav",
    "resemble_sts": "Divij_2min_Resemble_Clone_STS.wav",
}


def data_audio_path(key: str):
    filename = COMPARISON_AUDIO_FILES.get(key)
    if not filename:
        return None
    path = os.path.join(DATA_DIR, filename)
    return path if os.path.exists(path) else None


COMPARISON_ROWS = [
    [
        "ElevenLabs",
        "Instant clone: ~2 min tested; Pro clone: 30–120 min source.",
        "Ready in seconds.",
        "Good — strongest cloned TTS result.",
        "Usable, but less natural than TTS.",
    ],
    [
        "Cartesia",
        "Instant clone: ~10 sec tested; Pro clone: 30–120 min source.",
        "Ready in seconds.",
        "Good — quickest cloned TTS result.",
        "Usable, but less natural than TTS.",
    ],
    [
        "Resemble AI",
        "Minimum ~30 sec; tested 30 sec and 2 min inputs.",
        "Ready in seconds.",
        "30 sec clone weak; 2 min clone good.",
        "Weak similarity in STS.",
    ],
]

TEXT_SAMPLES = [
    "My voice confirms my identity for this verification test.",
    "This is a live voice changer demonstration using a cloned voice.",
    "The system converts my spoken words into the target voice.",
    "Please verify that this voice sample matches the enrolled speaker.",
    "Today I am testing speaker verification and synthetic voice detection.",
    "A short sentence is easier to convert quickly and clearly.",
    "The quick brown fox jumps over the lazy dog.",
    "Access granted after successful voice identity confirmation.",
]

RECORDING_PROMPTS = [
    "My voice confirms my identity for this verification test.",
    "I am speaking clearly for the voice conversion demo.",
    "Please verify this short recording against the enrolled speaker.",
    "This sentence is recorded first and then converted into the cloned voice.",
    "The system should transcribe this sentence and replay it in the target voice.",
    "I will speak one sentence and pause for the demo workflow.",
]

DEMO_SCRIPT = """Suggested demo flow:
1. Read one sentence clearly.
2. Pause for about one second.
3. Wait for the cloned voice output.
4. Continue with the next sentence.

Best practice: use short sentences and headphones to avoid duplicate feedback."""

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 mono

@dataclass
class AppState:
    running: bool = False
    worker_thread: Optional[threading.Thread] = None

    api_key: str = ""
    voice_id: str = ""
    model_id: str = "eleven_flash_v2_5"
    stability: float = 0.45
    similarity_boost: float = 0.90
    input_device_index: Optional[int] = None
    playback_cooldown_ms: int = 900

    live_status: str = "Idle"
    latest_audio_path: Optional[str] = None
    latest_transcript: str = ""
    last_timing: str = ""
    log: str = ""
    chunk_paths: List[str] = field(default_factory=list)
    merged_audio_path: Optional[str] = None

    # Prevent Gradio autoplay from replaying the same file on every timer tick.
    last_sent_audio_path: Optional[str] = None

    # Prevent the mic from capturing generated speech and creating duplicated loops.
    ignore_until: float = 0.0

    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = AppState()
PROCESS_LOCK = threading.Lock()


def set_status(message: str):
    with STATE.lock:
        STATE.live_status = message


def append_log(message: str):
    with STATE.lock:
        timestamp = time.strftime("%H:%M:%S")
        STATE.log += f"[{timestamp}] {message}\n"
        STATE.live_status = message


def get_api_key(api_key_input: str) -> str:
    api_key = (api_key_input or "").strip() or os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise gr.Error("Missing API key. Paste it in Settings or set ELEVENLABS_API_KEY in .env.")
    return api_key


def get_voice_id(voice_id_input: str) -> str:
    voice_id = (voice_id_input or "").strip() or os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    if not voice_id:
        raise gr.Error("Missing Voice ID. Paste it in Settings or set ELEVENLABS_VOICE_ID in .env.")
    return voice_id


def get_example_voice_id(example_voice_id_input: str) -> str:
    voice_id = (example_voice_id_input or "").strip() or os.getenv("ELEVENLABS_EXAMPLE_VOICE_ID", "").strip()
    if not voice_id:
        raise gr.Error(
            "Missing example recording voice ID. Set ELEVENLABS_EXAMPLE_VOICE_ID in .env "
            "or paste it in Settings. This should be a different/source voice from the target cloned voice."
        )
    return voice_id


def parse_device_index(value: str):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        idx = int(value)
        return idx if idx >= 0 else None
    except ValueError:
        raise gr.Error("Input device index must be a number, e.g. 1 or 5. Leave empty for system default.")


def list_audio_devices():
    try:
        return str(sd.query_devices())
    except Exception as e:
        return f"Failed to query devices: {e}"


def get_wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 1.0


def write_wav_from_pcm_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    with wave.open(out.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return out.name


def transcribe_audio_file(audio_path: str, api_key: str) -> str:
    url = f"{ELEVENLABS_BASE_URL}/speech-to-text"
    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            headers={"xi-api-key": api_key},
            files={"file": f},
            data={"model_id": "scribe_v2"},
            timeout=120,
        )
    if response.status_code != 200:
        raise RuntimeError(f"STT failed: {response.status_code} {response.text}")
    return response.json().get("text", "").strip()


def tts_to_wav_file(text: str, voice_id: str, api_key: str, model_id: str, stability: float, similarity_boost: float) -> str:
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream"
    response = requests.post(
        url,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        params={"output_format": "pcm_16000"},
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"TTS failed: {response.status_code} {response.text}")
    return write_wav_from_pcm_bytes(response.content, SAMPLE_RATE)


def merge_wav_files(wav_paths: List[str]) -> Optional[str]:
    if not wav_paths:
        return None
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output.close()
    with wave.open(output.name, "wb") as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(SAMPLE_RATE)
        for path in wav_paths:
            with wave.open(path, "rb") as in_wav:
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
            out_wav.writeframes(np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.int16).tobytes())
    return output.name


def random_text_sample():
    return random.choice(TEXT_SAMPLES)


def random_recording_prompt():
    return random.choice(RECORDING_PROMPTS)


def generate_example_recordings(api_key_input, example_voice_id_input, model_id, stability, similarity_boost):
    """Generate short source-voice example clips.

    These should use a DIFFERENT voice from the target cloned voice.
    Configure via ELEVENLABS_EXAMPLE_VOICE_ID in .env or the Settings field.
    """
    api_key = get_api_key(api_key_input)
    example_voice_id = get_example_voice_id(example_voice_id_input)
    selected = RECORDING_PROMPTS[:3]
    outputs = []
    for prompt in selected:
        outputs.append(
            tts_to_wav_file(
                prompt,
                example_voice_id,
                api_key,
                model_id,
                stability,
                similarity_boost,
            )
        )
    summary = (
        "Generated source/example recordings with ELEVENLABS_EXAMPLE_VOICE_ID:\n"
        + "\n".join(f"- {x}" for x in selected)
        + "\n\nExample 1 has also been loaded into the clip input, so you can click Convert Clip directly."
    )
    # Return audio outputs, stored states, summary, and load Example 1 into clip input.
    return outputs[0], outputs[1], outputs[2], outputs[0], outputs[1], outputs[2], summary, outputs[0]


def use_example_recording(path):
    if not path:
        raise gr.Error("Generate example recordings first.")
    return path


def check_api(api_key_input: str):
    api_key = get_api_key(api_key_input)
    response = requests.get(f"{ELEVENLABS_BASE_URL}/user", headers={"xi-api-key": api_key}, timeout=30)
    if response.status_code != 200:
        raise gr.Error(f"API check failed: {response.status_code}\n{response.text}")
    sub = response.json().get("subscription", {})
    return (
        "API key works.\n\n"
        f"Character count: {sub.get('character_count')}\n"
        f"Character limit: {sub.get('character_limit')}\n"
        f"Can extend character limit: {sub.get('can_extend_character_limit')}"
    )


def text_to_voice_once(text, api_key_input, voice_id_input, model_id, stability, similarity_boost):
    api_key = get_api_key(api_key_input)
    voice_id = get_voice_id(voice_id_input)
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please type text.")
    t0 = time.perf_counter()
    path = tts_to_wav_file(text, voice_id, api_key, model_id, stability, similarity_boost)
    t1 = time.perf_counter()
    return path, text, f"TTS: {t1 - t0:.2f}s"


def recorded_clip_to_cloned_voice(audio_path, api_key_input, voice_id_input, model_id, stability, similarity_boost):
    api_key = get_api_key(api_key_input)
    voice_id = get_voice_id(voice_id_input)
    if not audio_path:
        raise gr.Error("Please record or upload an audio clip first.")
    t0 = time.perf_counter()
    transcript = transcribe_audio_file(audio_path, api_key)
    t1 = time.perf_counter()
    if not transcript:
        raise gr.Error("No transcript detected from the clip.")
    output_path = tts_to_wav_file(transcript, voice_id, api_key, model_id, stability, similarity_boost)
    t2 = time.perf_counter()
    timing = f"STT: {t1 - t0:.2f}s | TTS: {t2 - t1:.2f}s | Total: {t2 - t0:.2f}s"
    return output_path, transcript, timing


def process_utterance(pcm_bytes: bytes):
    # Avoid old chunks finishing after new chunks, and prevent request overlap.
    if not PROCESS_LOCK.acquire(blocking=False):
        append_log("Already processing; overlapping chunk ignored.")
        return

    try:
        with STATE.lock:
            api_key = STATE.api_key
            voice_id = STATE.voice_id
            model_id = STATE.model_id
            stability = STATE.stability
            similarity_boost = STATE.similarity_boost
            playback_cooldown_ms = STATE.playback_cooldown_ms

        if not api_key or not voice_id:
            append_log("Missing API key or Voice ID; skipping utterance.")
            return

        total_start = time.perf_counter()
        input_wav = write_wav_from_pcm_bytes(pcm_bytes, SAMPLE_RATE)

        append_log("Speech ended → transcribing...")
        stt_start = time.perf_counter()
        transcript = transcribe_audio_file(input_wav, api_key)
        stt_end = time.perf_counter()

        if not transcript:
            append_log("No transcript detected; listening again.")
            return

        append_log(f"Transcript ready: {transcript}")
        append_log("Generating cloned voice...")

        tts_start = time.perf_counter()
        output_wav = tts_to_wav_file(transcript, voice_id, api_key, model_id, stability, similarity_boost)
        tts_end = time.perf_counter()

        output_duration = get_wav_duration_sec(output_wav)
        timing = (
            f"STT: {stt_end - stt_start:.2f}s | "
            f"TTS: {tts_end - tts_start:.2f}s | "
            f"Audio: {output_duration:.1f}s | "
            f"Total after speech end: {tts_end - total_start:.2f}s"
        )

        with STATE.lock:
            STATE.latest_audio_path = output_wav
            STATE.latest_transcript = transcript
            STATE.last_timing = timing
            STATE.chunk_paths.append(output_wav)
            STATE.ignore_until = time.time() + output_duration + (playback_cooldown_ms / 1000.0)
            STATE.live_status = "Cloned voice ready. Playback cooldown active..."

        append_log(f"Cloned voice ready. {timing}")

    except Exception as e:
        append_log(f"Error: {e}")
    finally:
        PROCESS_LOCK.release()


def microphone_worker(vad_aggressiveness: int, silence_ms: int, min_speech_ms: int, input_device_index: Optional[int]):
    audio_q = queue.Queue()
    vad = webrtcvad.Vad(int(vad_aggressiveness))
    silence_frames_limit = max(1, int(silence_ms / FRAME_MS))
    min_speech_frames = max(1, int(min_speech_ms / FRAME_MS))
    pre_roll_frames = int(250 / FRAME_MS)

    triggered = False
    voiced_frames = []
    pre_buffer = []
    silence_count = 0
    speech_started_at = None

    def callback(indata, frames, time_info, status):
        if status:
            append_log(f"Audio status: {status}")
        audio_q.put(bytes(indata))

    append_log(f"Microphone worker started. device={input_device_index if input_device_index is not None else 'default'}")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=CHANNELS,
            device=input_device_index,
            callback=callback,
        ):
            set_status("Listening... speak now")
            while True:
                with STATE.lock:
                    if not STATE.running:
                        break
                    ignore_until = STATE.ignore_until

                try:
                    frame = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if len(frame) != FRAME_BYTES:
                    continue

                if time.time() < ignore_until:
                    triggered = False
                    voiced_frames = []
                    pre_buffer = []
                    silence_count = 0
                    set_status(f"Playback cooldown... {max(0, ignore_until - time.time()):.1f}s")
                    continue

                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if not triggered:
                    pre_buffer.append(frame)
                    if len(pre_buffer) > pre_roll_frames:
                        pre_buffer.pop(0)
                    if is_speech:
                        triggered = True
                        speech_started_at = time.perf_counter()
                        voiced_frames = list(pre_buffer)
                        voiced_frames.append(frame)
                        silence_count = 0
                        append_log("Speech detected. Keep speaking...")
                else:
                    voiced_frames.append(frame)
                    if is_speech:
                        silence_count = 0
                        if speech_started_at:
                            set_status(f"Speaking... {time.perf_counter() - speech_started_at:.1f}s")
                    else:
                        silence_count += 1
                        remaining = max(0, silence_frames_limit - silence_count) * FRAME_MS
                        set_status(f"Pause detected. Finalising in ~{remaining} ms")

                    if silence_count >= silence_frames_limit:
                        speech_frame_count = len(voiced_frames) - silence_count
                        utterance_sec = len(voiced_frames) * FRAME_MS / 1000
                        if speech_frame_count >= min_speech_frames:
                            append_log(f"Utterance finalised ({utterance_sec:.1f}s). Sending to ElevenLabs...")
                            threading.Thread(target=process_utterance, args=(b"".join(voiced_frames),), daemon=True).start()
                        else:
                            append_log("Short noise ignored. Listening again...")

                        triggered = False
                        voiced_frames = []
                        pre_buffer = []
                        silence_count = 0
                        speech_started_at = None
                        set_status("Listening... speak now")

    except Exception as e:
        append_log(f"Microphone worker error: {e}")

    append_log("Microphone worker stopped.")


def start_continuous_mode(api_key_input, voice_id_input, model_id, stability, similarity_boost, vad_aggressiveness, silence_ms, min_speech_ms, input_device_index_text, playback_cooldown_ms):
    api_key = get_api_key(api_key_input)
    voice_id = get_voice_id(voice_id_input)
    device_index = parse_device_index(input_device_index_text)

    with STATE.lock:
        if STATE.running:
            return "Already running."
        STATE.api_key = api_key
        STATE.voice_id = voice_id
        STATE.model_id = model_id
        STATE.stability = stability
        STATE.similarity_boost = similarity_boost
        STATE.input_device_index = device_index
        STATE.playback_cooldown_ms = int(playback_cooldown_ms)
        STATE.running = True
        STATE.live_status = "Starting microphone..."
        STATE.latest_audio_path = None
        STATE.latest_transcript = ""
        STATE.last_timing = ""
        STATE.log = ""
        STATE.chunk_paths = []
        STATE.merged_audio_path = None
        STATE.last_sent_audio_path = None
        STATE.ignore_until = 0.0

        worker = threading.Thread(
            target=microphone_worker,
            args=(vad_aggressiveness, silence_ms, min_speech_ms, device_index),
            daemon=True,
        )
        STATE.worker_thread = worker
        worker.start()

    return "Continuous microphone mode started."


def stop_continuous_mode():
    with STATE.lock:
        STATE.running = False
        STATE.live_status = "Stopping..."
    append_log("Stop requested.")
    return "Stopping microphone mode..."


def poll_outputs():
    with STATE.lock:
        if STATE.latest_audio_path and STATE.latest_audio_path != STATE.last_sent_audio_path:
            audio_update = STATE.latest_audio_path
            STATE.last_sent_audio_path = STATE.latest_audio_path
        else:
            audio_update = gr.update()
        return (
            STATE.live_status,
            audio_update,
            STATE.latest_transcript,
            STATE.last_timing,
            STATE.merged_audio_path,
            STATE.log,
        )


def clear_chunks():
    with STATE.lock:
        STATE.latest_audio_path = None
        STATE.latest_transcript = ""
        STATE.last_timing = ""
        STATE.chunk_paths = []
        STATE.merged_audio_path = None
        STATE.last_sent_audio_path = None
        STATE.ignore_until = 0.0
        STATE.log = ""
        STATE.live_status = "Cleared"
    return "Cleared", None, "", "", None, ""


def build_merged_audio():
    with STATE.lock:
        paths = list(STATE.chunk_paths)
    merged = merge_wav_files(paths)
    with STATE.lock:
        STATE.merged_audio_path = merged
    return merged


def render_useful_links_html() -> str:
    blocks = []
    for group, links in LINK_GROUPS.items():
        items = "".join(
            f'<li><a href="{url}" target="_blank">{name}</a></li>' for name, url in links
        )
        blocks.append(f"<div class='link-card'><h3>{group}</h3><ul>{items}</ul></div>")
    return "<div class='link-grid'>" + "".join(blocks) + "</div>"


def render_market_apps_html() -> str:
    cards = []
    for app in MARKET_APP_ROWS:
        links = " · ".join(
            f'<a href="{url}" target="_blank">{label}</a>' for label, url in app["links"]
        )
        cards.append(
            f"""
            <div class="market-card">
              <div class="market-image-wrap"><img src="{app['image']}" alt="{app['name']} screenshot" /></div>
              <div class="market-content">
                <h3>{app['name']}</h3>
                <div class="market-platform">{app['platform']}</div>
                <p>{app['intro']}</p>
                <div class="market-links">{links}</div>
              </div>
            </div>
            """
        )
    return "<div class='market-grid'>" + "".join(cards) + "</div>"


def render_comparison_html() -> str:
    rank_classes = ["rank-one", "rank-two", "rank-three"]
    rows = []
    for i, row in enumerate(COMPARISON_ROWS):
        tool, input_req, speed, tts, sts = row
        rank = i + 1
        cls = rank_classes[i] if i < len(rank_classes) else ""
        rows.append(
            f"""
            <tr class="{cls}">
              <td><span class="rank-badge">#{rank}</span><strong>{tool}</strong></td>
              <td>{input_req}</td>
              <td>{speed}</td>
              <td>{tts}</td>
              <td>{sts}</td>
            </tr>
            """
        )
    return (
        "<div class='comparison-wrap'>"
        "<table class='comparison-table'>"
        "<thead><tr><th>Tool / ranking</th><th>Input</th><th>Output speed</th><th>TTS</th><th>STS</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "<p class='comparison-note'>Ranking reflects current demo suitability based on your hands-on tests: quality first, then speed and backup value.</p>"
        "</div>"
    )


CUSTOM_CSS = """
:root {
  --primary: #2563eb;
  --primary-soft: #dbeafe;
  --accent: #7c3aed;
  --ink: #111827;
  --muted: #6b7280;
  --panel: #ffffff;
  --panel-border: #e5e7eb;
}

html {
  overflow-y: scroll;
}

.gradio-container {
  width: 96vw !important;
  max-width: 1480px !important;
  min-width: 1080px !important;
  margin: 0 auto !important;
  background: #f8fafc !important;
}

@media (max-width: 1120px) {
  .gradio-container {
    min-width: auto !important;
    width: 98vw !important;
  }
}

#hero {
  padding: 24px 28px;
  border-radius: 22px;
  background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 45%, #f5f3ff 100%);
  border: 1px solid #dbeafe;
  color: var(--ink);
  margin-bottom: 16px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

#hero h1 {
  margin: 0 0 8px 0;
  font-size: 34px;
  line-height: 1.1;
  letter-spacing: -0.03em;
  color: #0f172a !important;
  font-weight: 800;
}

#hero p {
  margin: 0;
  color: #334155 !important;
  font-size: 16px;
}

.hero-actions {
  margin-top: 16px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.hero-button {
  display: inline-block;
  padding: 10px 14px;
  border-radius: 12px;
  background: #2563eb;
  color: #ffffff !important;
  text-decoration: none !important;
  font-weight: 700;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
}

.hero-button.secondary {
  background: #ffffff;
  color: #1d4ed8 !important;
  border: 1px solid #bfdbfe;
  box-shadow: none;
}

#hero .tagline {
  display: inline-block;
  margin-bottom: 10px;
  padding: 5px 10px;
  border-radius: 999px;
  background: #ffffffcc;
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  font-size: 13px;
  font-weight: 700;
}

.mode-note {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 10px;
}

.compact-card {
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 14px;
  background: var(--panel);
}

button.primary, .primary button {
  border-radius: 12px !important;
}

textarea, input, .wrap, .block {
  border-radius: 12px !important;
}

.tabitem {
  background: #ffffff !important;
  border-radius: 0 0 18px 18px !important;
  padding: 12px !important;
}

.panel-soft {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 10px;
}

/* Useful links */
.link-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.link-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 14px 16px;
  min-height: 130px;
}
.link-card h3 { margin: 0 0 8px 0; font-size: 15px; color: #0f172a; }
.link-card ul { padding-left: 18px; margin: 0; }
.link-card li { margin: 6px 0; }
.link-card a { color: #2563eb !important; font-weight: 600; text-decoration: none !important; }

/* Market app cards */
.market-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  align-items: stretch;
}
.market-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
  min-height: 610px;
}
.market-image-wrap {
  height: 430px;
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  padding: 20px;
}
.market-image-wrap img {
  width: 220px;
  height: 390px;
  object-fit: contain;
  object-position: center;
  padding: 10px;
  border-radius: 26px;
  background: #ffffff;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
  border: 1px solid #e5e7eb;
}
.market-content {
  padding: 14px 16px 16px 16px;
  display: flex;
  flex-direction: column;
  flex: 1;
}
.market-content h3 { margin: 0 0 4px 0; color: #0f172a; font-size: 17px; }
.market-platform {
  display: inline-block;
  margin-bottom: 8px;
  padding: 3px 8px;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 700;
}
.market-content p {
  color: #475569;
  font-size: 13px;
  line-height: 1.45;
  flex: 1;
}
.market-links {
  padding-top: 8px;
  border-top: 1px solid #f1f5f9;
}
.market-links a { color: #2563eb !important; font-weight: 700; text-decoration: none !important; }

/* Ranking comparison table */
.comparison-wrap {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}
.comparison-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.comparison-table th {
  background: #0f172a;
  color: #ffffff;
  text-align: left;
  padding: 12px 14px;
  font-weight: 700;
}
.comparison-table td {
  padding: 13px 14px;
  border-top: 1px solid #e5e7eb;
  vertical-align: top;
  color: #111827;
}
.comparison-table tr.rank-one td { background: #ecfdf5; }
.comparison-table tr.rank-two td { background: #eff6ff; }
.comparison-table tr.rank-three td { background: #fff7ed; }
.comparison-table tr.rank-one td:first-child { border-left: 6px solid #10b981; }
.comparison-table tr.rank-two td:first-child { border-left: 6px solid #3b82f6; }
.comparison-table tr.rank-three td:first-child { border-left: 6px solid #f97316; }
.rank-badge {
  display: inline-block;
  margin-right: 8px;
  padding: 3px 8px;
  border-radius: 999px;
  background: #111827;
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
}
.comparison-note {
  margin: 0;
  padding: 10px 14px 12px 14px;
  color: #64748b;
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  font-size: 13px;
}

@media (max-width: 1180px) {
  .link-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .market-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .link-grid { grid-template-columns: 1fr; }
  .market-grid { grid-template-columns: 1fr; }
}

"""

with gr.Blocks(title="Voice Changer Lab", css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <div id="hero">
          <div class="tagline">Local Demo · ElevenLabs Cloned Voice</div>
          <h1>Voice Changer Lab</h1>
          <p>Convert text, recorded clips, or live microphone speech into your cloned voice.</p>
          <div class="hero-actions">
            <a class="hero-button" href="https://elevenlabs.io/app/voice-lab" target="_blank">Open ElevenLabs Voice Lab</a>
            <a class="hero-button secondary" href="#" onclick="document.getElementById('useful-links-anchor').scrollIntoView({behavior:'smooth', block:'start'}); return false;">Useful Links</a>
          </div>
        </div>
        """
    )

    with gr.Accordion("Settings", open=False):
        gr.Markdown("Use `.env` for a cleaner demo: `ELEVENLABS_API_KEY=...` and optional `ELEVENLABS_VOICE_ID=...`.")
        with gr.Row():
            api_key_input = gr.Textbox(label="ElevenLabs API Key", placeholder="Leave empty if using .env", type="password")
            voice_id_input = gr.Textbox(label="Target Voice ID", placeholder="Leave empty if using ELEVENLABS_VOICE_ID in .env")
            example_voice_id_input = gr.Textbox(label="Example Source Voice ID", placeholder="Leave empty if using ELEVENLABS_EXAMPLE_VOICE_ID in .env")
        with gr.Row():
            model_id_input = gr.Dropdown(
                label="TTS Model",
                choices=["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"],
                value="eleven_flash_v2_5",
            )
            stability_input = gr.Slider(label="Stability", minimum=0.0, maximum=1.0, value=0.45, step=0.05)
            similarity_input = gr.Slider(label="Similarity Boost", minimum=0.0, maximum=1.0, value=0.90, step=0.05)
        with gr.Accordion("API test", open=False):
            check_btn = gr.Button("Check API Key")
            api_status = gr.Textbox(label="API Status", lines=4)
            check_btn.click(fn=check_api, inputs=[api_key_input], outputs=[api_status])

    with gr.Tabs():
        with gr.Tab("1. Text Voice Changer"):
            gr.Markdown("Type text and generate it using the cloned voice. Use a sample when you want a quick clean demo sentence.", elem_classes="mode-note")
            with gr.Row():
                random_text_btn = gr.Button("Random Sample")
                clear_text_btn = gr.Button("Clear Text")
            text_input = gr.Textbox(label="Text", lines=4, placeholder="Type a sentence or click Random Sample...")
            text_btn = gr.Button("Generate Voice", variant="primary")
            text_audio = gr.Audio(label="Output", type="filepath", autoplay=True)
            text_used = gr.Textbox(label="Text Used", lines=3)
            text_timing = gr.Textbox(label="Timing", lines=1)
            random_text_btn.click(fn=random_text_sample, inputs=[], outputs=[text_input])
            clear_text_btn.click(fn=lambda: "", inputs=[], outputs=[text_input])
            text_btn.click(
                fn=text_to_voice_once,
                inputs=[text_input, api_key_input, voice_id_input, model_id_input, stability_input, similarity_input],
                outputs=[text_audio, text_used, text_timing],
            )

        with gr.Tab("2. Recorded Clip Voice Changer"):
            gr.Markdown("Use a prompt as the source sentence, record it or upload an example clip, then convert the spoken content into the cloned voice.", elem_classes="mode-note")
            with gr.Row():
                random_prompt_btn = gr.Button("Random Recording Prompt")
                generate_examples_btn = gr.Button("Generate Example Recordings")
            recording_prompt = gr.Textbox(
                label="Prompt to read aloud",
                lines=3,
                value="My voice confirms my identity for this verification test.",
                interactive=True,
            )
            with gr.Accordion("Example recording prompts", open=False):
                gr.Examples(
                    examples=[[p] for p in RECORDING_PROMPTS],
                    inputs=[recording_prompt],
                    label="Click a prompt to use it",
                )
            with gr.Accordion("Generated example recordings", open=False):
                gr.Markdown("These are source/example clips generated from `ELEVENLABS_EXAMPLE_VOICE_ID`. Example 1 is automatically loaded into the clip input after generation.")
                example_path_1 = gr.State()
                example_path_2 = gr.State()
                example_path_3 = gr.State()
                with gr.Row():
                    example_audio_1 = gr.Audio(label="Example 1", type="filepath")
                    example_audio_2 = gr.Audio(label="Example 2", type="filepath")
                    example_audio_3 = gr.Audio(label="Example 3", type="filepath")
                with gr.Row():
                    use_example_1_btn = gr.Button("Use Example 1")
                    use_example_2_btn = gr.Button("Use Example 2")
                    use_example_3_btn = gr.Button("Use Example 3")
                example_summary = gr.Textbox(label="Generated examples", lines=5)
            clip_input = gr.Audio(label="Record or upload voice clip", sources=["microphone", "upload"], type="filepath")
            clip_btn = gr.Button("Convert Clip", variant="primary")
            clip_audio_output = gr.Audio(label="Output", type="filepath", autoplay=True)
            clip_transcript_output = gr.Textbox(label="Transcript", lines=4)
            clip_timing_output = gr.Textbox(label="Timing", lines=1)
            random_prompt_btn.click(fn=random_recording_prompt, inputs=[], outputs=[recording_prompt])
            generate_examples_btn.click(
                fn=generate_example_recordings,
                inputs=[api_key_input, example_voice_id_input, model_id_input, stability_input, similarity_input],
                outputs=[
                    example_audio_1,
                    example_audio_2,
                    example_audio_3,
                    example_path_1,
                    example_path_2,
                    example_path_3,
                    example_summary,
                    clip_input,
                ],
            )
            use_example_1_btn.click(fn=use_example_recording, inputs=[example_path_1], outputs=[clip_input])
            use_example_2_btn.click(fn=use_example_recording, inputs=[example_path_2], outputs=[clip_input])
            use_example_3_btn.click(fn=use_example_recording, inputs=[example_path_3], outputs=[clip_input])
            clip_btn.click(
                fn=recorded_clip_to_cloned_voice,
                inputs=[clip_input, api_key_input, voice_id_input, model_id_input, stability_input, similarity_input],
                outputs=[clip_audio_output, clip_transcript_output, clip_timing_output],
            )

        with gr.Tab("3. Continuous Mic Voice Changer"):
            gr.Markdown("1. Start mic. 2. Speak one short sentence. 3. Pause and wait for cloned output. Use headphones to avoid feedback.", elem_classes="mode-note")
            with gr.Accordion("Advanced microphone settings", open=False):
                with gr.Row():
                    device_index_input = gr.Textbox(
                        label="Input Device Index",
                        value="",
                        placeholder="Leave empty for default. Example: 1 or 5.",
                    )
                    list_devices_btn = gr.Button("List Devices")
                devices_box = gr.Textbox(label="Audio Devices", lines=8)
                list_devices_btn.click(fn=list_audio_devices, inputs=[], outputs=[devices_box])
                with gr.Row():
                    vad_input = gr.Slider(label="VAD Aggressiveness", minimum=0, maximum=3, value=2, step=1)
                    silence_input = gr.Slider(label="Speech End Silence ms", minimum=300, maximum=2000, value=800, step=100)
                    min_speech_input = gr.Slider(label="Minimum Speech ms", minimum=300, maximum=3000, value=700, step=100)
                    playback_cooldown_input = gr.Slider(label="Playback Cooldown ms", minimum=0, maximum=3000, value=900, step=100)
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop")
                clear_btn = gr.Button("Clear")
            status_output = gr.Textbox(label="Status", lines=1)
            live_status_output = gr.Textbox(label="Live Status", lines=1)
            latest_audio = gr.Audio(label="Latest Output", type="filepath", autoplay=True)
            with gr.Row():
                latest_transcript = gr.Textbox(label="Latest Transcript", lines=3)
                timing_output = gr.Textbox(label="Latest Timing", lines=3)
            with gr.Accordion("Session audio", open=False):
                with gr.Row():
                    build_merge_btn = gr.Button("Build / Update Merged Audio")
                    merged_audio = gr.Audio(label="Merged Final Audio", type="filepath")
                log_output = gr.Textbox(label="Log", lines=12)
                refresh_btn = gr.Button("Refresh Now")

            start_btn.click(
                fn=start_continuous_mode,
                inputs=[
                    api_key_input,
                    voice_id_input,
                    model_id_input,
                    stability_input,
                    similarity_input,
                    vad_input,
                    silence_input,
                    min_speech_input,
                    device_index_input,
                    playback_cooldown_input,
                ],
                outputs=[status_output],
            )
            stop_btn.click(fn=stop_continuous_mode, inputs=[], outputs=[status_output])
            clear_btn.click(
                fn=clear_chunks,
                inputs=[],
                outputs=[live_status_output, latest_audio, latest_transcript, timing_output, merged_audio, log_output],
            )
            refresh_btn.click(
                fn=poll_outputs,
                inputs=[],
                outputs=[live_status_output, latest_audio, latest_transcript, timing_output, merged_audio, log_output],
            )
            build_merge_btn.click(fn=build_merged_audio, inputs=[], outputs=[merged_audio])
            timer = gr.Timer(0.5)
            timer.tick(
                fn=poll_outputs,
                inputs=[],
                outputs=[live_status_output, latest_audio, latest_transcript, timing_output, merged_audio, log_output],
            )


    with gr.Accordion("Apps on markets", open=False):
        gr.Markdown("Mobile app references for consumer-style voice changer UX and screenshots.")
        gr.HTML(render_market_apps_html())

    gr.HTML('<div id="useful-links-anchor" style="height: 1px;"></div>')
    with gr.Accordion("Useful links", open=False):
        gr.HTML(render_useful_links_html())

    with gr.Accordion("Model / audio input performance comparison", open=False):
        gr.Markdown(
            "Concise comparison based on current hands-on tests. Use the audio players below for side-by-side playback."
        )
        gr.HTML(render_comparison_html())
        gr.Markdown("### Output audio comparison")
        gr.Markdown(
            "The comparison players below load from the local `data/` folder when the files exist. "
            "You can also upload/replace files manually. Expected folder: `./data/`."
        )
        with gr.Row():
            eleven_tts_audio = gr.Audio(
                value=data_audio_path("eleven_tts"),
                label="ElevenLabs TTS · Divij_2min_ElevenLabs_InstantClone_TTS.mp3",
                sources=["upload"],
                type="filepath",
            )
            eleven_sts_audio = gr.Audio(
                value=data_audio_path("eleven_sts"),
                label="ElevenLabs STS · Divij_2min_ElevenLabs_InstantClone_STS.mp3",
                sources=["upload"],
                type="filepath",
            )
        with gr.Row():
            cartesia_tts_audio = gr.Audio(
                value=data_audio_path("cartesia_tts"),
                label="Cartesia TTS · Divij_10sec_Cartesia_InstantClone_TTS.wav",
                sources=["upload"],
                type="filepath",
            )
            cartesia_sts_audio = gr.Audio(
                value=data_audio_path("cartesia_sts"),
                label="Cartesia STS · Divij_10sec_Cartesia_InstantClone_STS.mp3",
                sources=["upload"],
                type="filepath",
            )
        with gr.Row():
            resemble_tts_audio = gr.Audio(
                value=data_audio_path("resemble_tts"),
                label="Resemble TTS · Divij_2min_Resemble_Clone_TTS.wav",
                sources=["upload"],
                type="filepath",
            )
            resemble_sts_audio = gr.Audio(
                value=data_audio_path("resemble_sts"),
                label="Resemble STS · Divij_2min_Resemble_Clone_STS.wav",
                sources=["upload"],
                type="filepath",
            )



demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
