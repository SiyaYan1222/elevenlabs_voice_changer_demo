import os
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

    latest_audio_path: Optional[str] = None
    latest_transcript: str = ""
    log: str = ""
    chunk_paths: List[str] = field(default_factory=list)
    merged_audio_path: Optional[str] = None

    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = AppState()


def get_api_key(api_key_input: str) -> str:
    api_key = (api_key_input or "").strip() or os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise gr.Error("Missing API key. Paste it in the UI or set ELEVENLABS_API_KEY in .env.")
    return api_key


def append_log(message: str):
    with STATE.lock:
        timestamp = time.strftime("%H:%M:%S")
        STATE.log += f"[{timestamp}] {message}\n"


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

    data = response.json()
    text = data.get("text", "").strip()
    return text


def tts_to_wav_file(
    text: str,
    voice_id: str,
    api_key: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
) -> str:
    """
    Request raw PCM output from ElevenLabs and wrap it into a WAV file.
    This makes merging chunks easier than MP3 concatenation.
    """
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream"

    response = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
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
            try:
                with wave.open(path, "rb") as in_wav:
                    frames = in_wav.readframes(in_wav.getnframes())
                    out_wav.writeframes(frames)

                # Add 300 ms silence between chunks
                silence = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.int16).tobytes()
                out_wav.writeframes(silence)
            except Exception as e:
                append_log(f"Merge skipped a bad chunk: {e}")

    return output.name


def check_api(api_key_input: str):
    api_key = get_api_key(api_key_input)

    response = requests.get(
        f"{ELEVENLABS_BASE_URL}/user",
        headers={"xi-api-key": api_key},
        timeout=30,
    )

    if response.status_code != 200:
        raise gr.Error(f"API check failed: {response.status_code}\n{response.text}")

    data = response.json()
    sub = data.get("subscription", {})

    return (
        "API key works.\n\n"
        f"Character count: {sub.get('character_count')}\n"
        f"Character limit: {sub.get('character_limit')}\n"
        f"Can extend character limit: {sub.get('can_extend_character_limit')}"
    )


def text_to_voice_once(text, api_key_input, voice_id, model_id, stability, similarity_boost):
    api_key = get_api_key(api_key_input)
    voice_id = (voice_id or "").strip()
    text = (text or "").strip()

    if not voice_id:
        raise gr.Error("Please paste Voice ID.")
    if not text:
        raise gr.Error("Please type text.")

    path = tts_to_wav_file(
        text=text,
        voice_id=voice_id,
        api_key=api_key,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )

    return path, text


def recorded_clip_to_cloned_voice(
    audio_path,
    api_key_input,
    voice_id,
    model_id,
    stability,
    similarity_boost,
):
    """
    Recorded/uploaded clip → ElevenLabs STT → transcript → ElevenLabs TTS cloned voice.
    """
    api_key = get_api_key(api_key_input)
    voice_id = (voice_id or "").strip()

    if not audio_path:
        raise gr.Error("Please record or upload an audio clip first.")

    if not voice_id:
        raise gr.Error("Please paste Voice ID.")

    transcript = transcribe_audio_file(audio_path, api_key)

    if not transcript:
        raise gr.Error("No transcript detected from the clip.")

    output_path = tts_to_wav_file(
        text=transcript,
        voice_id=voice_id,
        api_key=api_key,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )

    return output_path, transcript


def process_utterance(pcm_bytes: bytes):
    """
    Called whenever VAD detects the end of one spoken sentence/utterance.
    """
    with STATE.lock:
        api_key = STATE.api_key
        voice_id = STATE.voice_id
        model_id = STATE.model_id
        stability = STATE.stability
        similarity_boost = STATE.similarity_boost

    if not api_key or not voice_id:
        append_log("Missing API key or Voice ID; skipping utterance.")
        return

    try:
        input_wav = write_wav_from_pcm_bytes(pcm_bytes, SAMPLE_RATE)

        append_log("Speech ended. Transcribing...")
        transcript = transcribe_audio_file(input_wav, api_key)

        if not transcript:
            append_log("No transcript detected; skipping.")
            return

        append_log(f"Transcript: {transcript}")
        append_log("Generating cloned voice...")

        output_wav = tts_to_wav_file(
            text=transcript,
            voice_id=voice_id,
            api_key=api_key,
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
        )

        with STATE.lock:
            STATE.latest_audio_path = output_wav
            STATE.latest_transcript = transcript
            STATE.chunk_paths.append(output_wav)
            STATE.merged_audio_path = merge_wav_files(STATE.chunk_paths)

        append_log("Cloned voice chunk ready.")

    except Exception as e:
        append_log(f"Error: {e}")


def microphone_worker(
    vad_aggressiveness: int,
    silence_ms: int,
    min_speech_ms: int,
):
    """
    Continuously captures local microphone audio.
    Uses WebRTC VAD to detect speech start/end.
    """
    audio_q = queue.Queue()
    vad = webrtcvad.Vad(int(vad_aggressiveness))

    silence_frames_limit = max(1, int(silence_ms / FRAME_MS))
    min_speech_frames = max(1, int(min_speech_ms / FRAME_MS))
    pre_roll_frames = int(300 / FRAME_MS)

    triggered = False
    voiced_frames = []
    pre_buffer = []
    silence_count = 0

    def callback(indata, frames, time_info, status):
        if status:
            append_log(f"Audio status: {status}")
        audio_q.put(bytes(indata))

    append_log("Microphone worker started.")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=CHANNELS,
            callback=callback,
        ):
            while True:
                with STATE.lock:
                    if not STATE.running:
                        break

                try:
                    frame = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if len(frame) != FRAME_BYTES:
                    continue

                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if not triggered:
                    pre_buffer.append(frame)
                    if len(pre_buffer) > pre_roll_frames:
                        pre_buffer.pop(0)

                    if is_speech:
                        triggered = True
                        voiced_frames = list(pre_buffer)
                        voiced_frames.append(frame)
                        silence_count = 0
                        append_log("Speech started.")
                else:
                    voiced_frames.append(frame)

                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1

                    if silence_count >= silence_frames_limit:
                        speech_frame_count = len(voiced_frames) - silence_count

                        if speech_frame_count >= min_speech_frames:
                            utterance_pcm = b"".join(voiced_frames)
                            threading.Thread(
                                target=process_utterance,
                                args=(utterance_pcm,),
                                daemon=True,
                            ).start()
                        else:
                            append_log("Short noise ignored.")

                        triggered = False
                        voiced_frames = []
                        pre_buffer = []
                        silence_count = 0

    except Exception as e:
        append_log(f"Microphone worker error: {e}")

    append_log("Microphone worker stopped.")


def start_continuous_mode(
    api_key_input,
    voice_id,
    model_id,
    stability,
    similarity_boost,
    vad_aggressiveness,
    silence_ms,
    min_speech_ms,
):
    api_key = get_api_key(api_key_input)
    voice_id = (voice_id or "").strip()

    if not voice_id:
        raise gr.Error("Please paste your ElevenLabs Voice ID.")

    with STATE.lock:
        if STATE.running:
            return "Already running."

        STATE.api_key = api_key
        STATE.voice_id = voice_id
        STATE.model_id = model_id
        STATE.stability = stability
        STATE.similarity_boost = similarity_boost
        STATE.running = True
        STATE.latest_audio_path = None
        STATE.latest_transcript = ""
        STATE.log = ""
        STATE.chunk_paths = []
        STATE.merged_audio_path = None

        worker = threading.Thread(
            target=microphone_worker,
            args=(vad_aggressiveness, silence_ms, min_speech_ms),
            daemon=True,
        )
        STATE.worker_thread = worker
        worker.start()

    return "Continuous microphone mode started."


def stop_continuous_mode():
    with STATE.lock:
        STATE.running = False

    append_log("Stop requested.")
    return "Stopping microphone mode..."


def poll_outputs():
    with STATE.lock:
        return (
            STATE.latest_audio_path,
            STATE.latest_transcript,
            STATE.merged_audio_path,
            STATE.log,
        )


def clear_chunks():
    with STATE.lock:
        STATE.latest_audio_path = None
        STATE.latest_transcript = ""
        STATE.chunk_paths = []
        STATE.merged_audio_path = None
        STATE.log = ""

    return None, "", None, ""


with gr.Blocks(title="ElevenLabs Continuous Voice Changer Demo") as demo:
    gr.Markdown(
        """
        # ElevenLabs Continuous Cloned Voice Demo

        Modes:
        1. **Text Test** — quick model/voice quality check.
        2. **Recorded Clip** — record/upload audio clip, transcribe, then output cloned voice.
        3. **Continuous Microphone** — listen locally, auto-detect speech end, then output cloned voice sentence by sentence.

        For the API key, paste it below or set `ELEVENLABS_API_KEY` in `.env`.
        """
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="ElevenLabs API Key Optional",
            placeholder="Leave empty if using .env",
            type="password",
        )
        voice_id_input = gr.Textbox(
            label="Voice ID",
            placeholder="Paste cloned voice_id here",
        )

    with gr.Row():
        model_id_input = gr.Dropdown(
            label="TTS Model",
            choices=[
                "eleven_flash_v2_5",
                "eleven_turbo_v2_5",
                "eleven_multilingual_v2",
            ],
            value="eleven_flash_v2_5",
        )
        stability_input = gr.Slider(
            label="Stability",
            minimum=0.0,
            maximum=1.0,
            value=0.45,
            step=0.05,
        )
        similarity_input = gr.Slider(
            label="Similarity Boost",
            minimum=0.0,
            maximum=1.0,
            value=0.90,
            step=0.05,
        )

    check_btn = gr.Button("Check API Key")
    api_status = gr.Textbox(label="API Status", lines=4)
    check_btn.click(fn=check_api, inputs=[api_key_input], outputs=[api_status])

    with gr.Tab("Text Test"):
        text_input = gr.Textbox(
            label="Text",
            lines=3,
            placeholder="Type a sentence to test the cloned voice...",
        )
        text_btn = gr.Button("Generate")
        text_audio = gr.Audio(label="Text TTS Output", type="filepath", autoplay=True)
        text_used = gr.Textbox(label="Text Used", lines=3)

        text_btn.click(
            fn=text_to_voice_once,
            inputs=[
                text_input,
                api_key_input,
                voice_id_input,
                model_id_input,
                stability_input,
                similarity_input,
            ],
            outputs=[text_audio, text_used],
        )

    with gr.Tab("Recorded Clip"):
        gr.Markdown(
            """
            Record a short clip or upload an audio file. The app will:

            `audio clip → STT transcript → cloned voice TTS output`
            """
        )

        clip_input = gr.Audio(
            label="Record or upload voice clip",
            sources=["microphone", "upload"],
            type="filepath",
        )

        clip_btn = gr.Button("Generate Cloned Voice from Clip")

        clip_audio_output = gr.Audio(
            label="Cloned Voice Output",
            type="filepath",
            autoplay=True,
        )

        clip_transcript_output = gr.Textbox(
            label="Transcript",
            lines=4,
        )

        clip_btn.click(
            fn=recorded_clip_to_cloned_voice,
            inputs=[
                clip_input,
                api_key_input,
                voice_id_input,
                model_id_input,
                stability_input,
                similarity_input,
            ],
            outputs=[
                clip_audio_output,
                clip_transcript_output,
            ],
        )

    with gr.Tab("Continuous Microphone"):
        gr.Markdown(
            """
            ### How to use

            1. Click **Start Continuous Mic**
            2. Speak one sentence
            3. Pause briefly
            4. The cloned voice output should appear and autoplay
            5. Repeat
            6. Click **Stop**
            7. Download merged audio if needed
            """
        )

        with gr.Row():
            vad_input = gr.Slider(
                label="VAD Aggressiveness",
                minimum=0,
                maximum=3,
                value=2,
                step=1,
                info="Higher = stricter speech detection. Try 2 first.",
            )
            silence_input = gr.Slider(
                label="Speech End Silence ms",
                minimum=300,
                maximum=2000,
                value=900,
                step=100,
                info="How long silence before one sentence is finalised.",
            )
            min_speech_input = gr.Slider(
                label="Minimum Speech ms",
                minimum=300,
                maximum=3000,
                value=700,
                step=100,
                info="Ignore noise shorter than this.",
            )

        with gr.Row():
            start_btn = gr.Button("Start Continuous Mic", variant="primary")
            stop_btn = gr.Button("Stop")
            clear_btn = gr.Button("Clear Chunks")

        status_output = gr.Textbox(label="Status", lines=1)

        latest_audio = gr.Audio(
            label="Latest Cloned Voice Chunk",
            type="filepath",
            autoplay=True,
        )

        latest_transcript = gr.Textbox(label="Latest Transcript", lines=3)

        merged_audio = gr.Audio(label="Merged Final Audio", type="filepath")

        log_output = gr.Textbox(label="Log", lines=14)

        refresh_btn = gr.Button("Refresh Output / Log")

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
            ],
            outputs=[status_output],
        )

        stop_btn.click(fn=stop_continuous_mode, inputs=[], outputs=[status_output])

        refresh_btn.click(
            fn=poll_outputs,
            inputs=[],
            outputs=[latest_audio, latest_transcript, merged_audio, log_output],
        )

        clear_btn.click(
            fn=clear_chunks,
            inputs=[],
            outputs=[latest_audio, latest_transcript, merged_audio, log_output],
        )

        # Auto-refresh every 1 second.
        # If your Gradio version does not support Timer, comment out this block
        # and use the "Refresh Output / Log" button instead.
        timer = gr.Timer(1)
        timer.tick(
            fn=poll_outputs,
            inputs=[],
            outputs=[latest_audio, latest_transcript, merged_audio, log_output],
        )


demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
