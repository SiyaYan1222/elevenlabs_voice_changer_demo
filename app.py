import os
import tempfile
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


def get_api_key(api_key_input: str) -> str:
    """
    Use pasted API key first.
    If empty, fallback to ELEVENLABS_API_KEY from .env.
    """
    api_key = (api_key_input or "").strip() or os.getenv("ELEVENLABS_API_KEY", "").strip()

    if not api_key:
        raise gr.Error(
            "Missing ElevenLabs API key. Paste it in the UI or set ELEVENLABS_API_KEY in .env."
        )

    return api_key


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


def text_to_cloned_voice(
    text: str,
    voice_id: str,
    api_key_input: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
):
    api_key = get_api_key(api_key_input)
    voice_id = (voice_id or "").strip()
    text = (text or "").strip()

    if not voice_id:
        raise gr.Error("Please paste your ElevenLabs Voice ID.")

    if not text:
        raise gr.Error("Please type some text first.")

    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream"

    params = {
        "output_format": "mp3_44100_128"
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }

    response = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        params=params,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise gr.Error(f"TTS failed: {response.status_code}\n{response.text}")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    output_file.write(response.content)
    output_file.close()

    return output_file.name, text


def transcribe_audio(audio_path: str, api_key: str) -> str:
    if not audio_path:
        raise gr.Error("Please record or upload an audio file first.")

    url = f"{ELEVENLABS_BASE_URL}/speech-to-text"

    with open(audio_path, "rb") as audio_file:
        files = {
            "file": audio_file,
        }

        data = {
            "model_id": "scribe_v2",
        }

        response = requests.post(
            url,
            headers={
                "xi-api-key": api_key,
            },
            files=files,
            data=data,
            timeout=120,
        )

    if response.status_code != 200:
        raise gr.Error(f"Speech-to-text failed: {response.status_code}\n{response.text}")

    result = response.json()
    text = result.get("text", "").strip()

    if not text:
        raise gr.Error(f"No transcript returned.\nRaw response:\n{result}")

    return text


def mic_to_cloned_voice(
    audio_path: str,
    voice_id: str,
    api_key_input: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
):
    api_key = get_api_key(api_key_input)

    transcript = transcribe_audio(audio_path, api_key)

    audio_output, _ = text_to_cloned_voice(
        text=transcript,
        voice_id=voice_id,
        api_key_input=api_key,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )

    return audio_output, transcript


with gr.Blocks(title="ElevenLabs Cloned Voice Changer Demo") as demo:
    gr.Markdown(
        """
        # ElevenLabs Cloned Voice Changer Demo

        Simple local demo:

        - Type text → cloned voice output
        - Record microphone → transcribe → cloned voice output
        - Output audio can be played and downloaded

        For the API key, either paste it below or set `ELEVENLABS_API_KEY` in `.env`.
        """
    )

    with gr.Row():
        voice_id = gr.Textbox(
            label="Voice ID",
            placeholder="Paste your cloned ElevenLabs voice_id here",
        )

        api_key_input = gr.Textbox(
            label="ElevenLabs API Key (Optional)",
            placeholder="Leave empty if using .env",
            type="password",
        )

    with gr.Row():
        model_id = gr.Dropdown(
            label="TTS Model",
            choices=[
                "eleven_flash_v2_5",
                "eleven_turbo_v2_5",
                "eleven_multilingual_v2",
            ],
            value="eleven_flash_v2_5",
        )
        stability = gr.Slider(
            label="Stability",
            minimum=0.0,
            maximum=1.0,
            value=0.45,
            step=0.05,
        )
        similarity_boost = gr.Slider(
            label="Similarity Boost",
            minimum=0.0,
            maximum=1.0,
            value=0.90,
            step=0.05,
        )

    check_btn = gr.Button("Check API Key")
    api_status = gr.Textbox(label="API Status", lines=4)
    check_btn.click(
        fn=check_api,
        inputs=[api_key_input],
        outputs=[api_status],
    )

    gr.Markdown("---")

    with gr.Tab("Text Input"):
        text_input = gr.Textbox(
            label="Text to speak",
            placeholder="Type something here...",
            lines=4,
        )
        text_btn = gr.Button("Generate Cloned Voice from Text")
        text_audio_output = gr.Audio(
            label="Cloned Voice Output",
            type="filepath",
            autoplay=True,
        )
        text_transcript_output = gr.Textbox(label="Text Used", lines=3)

        text_btn.click(
            fn=text_to_cloned_voice,
            inputs=[
                text_input,
                voice_id,
                api_key_input,
                model_id,
                stability,
                similarity_boost,
            ],
            outputs=[
                text_audio_output,
                text_transcript_output,
            ],
        )

    with gr.Tab("Microphone Input"):
        mic_input = gr.Audio(
            label="Record microphone input",
            sources=["microphone"],
            type="filepath",
        )
        mic_btn = gr.Button("Convert Recording to Cloned Voice")
        mic_audio_output = gr.Audio(
            label="Cloned Voice Output",
            type="filepath",
            autoplay=True,
        )
        transcript_output = gr.Textbox(label="Transcript", lines=4)

        mic_btn.click(
            fn=mic_to_cloned_voice,
            inputs=[
                mic_input,
                voice_id,
                api_key_input,
                model_id,
                stability,
                similarity_boost,
            ],
            outputs=[
                mic_audio_output,
                transcript_output,
            ],
        )

demo.launch(server_name="0.0.0.0", server_port=7860)